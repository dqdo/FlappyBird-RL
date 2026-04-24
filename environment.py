"""
environment.py
DynamicFlappyEnv - Flappy Bird where pipe speed and gap size oscillate mid-game.
Also exposes a FixedFlappyEnv convenience wrapper for static configurations.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

# screen dimensions in pixels
SCREEN_W = 400
SCREEN_H = 600

# physics constants for bird movement
BASE_GRAVITY = 0.5
BASE_FLAP_VEL = -9.0

# base horizontal speed pipes move per frame
BASE_SPEED = 3.0

# bird's fixed horizontal position and collision radius
BIRD_X = 80
BIRD_RADIUS = 12

# pipe width and the horizontal gap between consecutive pipes
PIPE_W = 60
PIPE_SPACING = 320

# frames per second for the game loop
FPS = 60


# dynamic environment where pipe speed and gap size oscillate over time
class DynamicFlappyEnv:
    # observation is 7 normalised floats: bird_y, bird_vel, dist_to_pipe,
    # gap_centre_y, gap_size, current_speed_norm, gap_phase_norm
    observation_space_size = 7
    # two actions: 0 = do nothing, 1 = flap
    action_space_size = 2

    # set up all oscillation parameters and optionally open a render window
    def __init__(
        self,
        speed_min: float = 0.75, # slowest speed multiplier
        speed_max: float = 1.25, # fastest speed multiplier
        gap_min: int = 150, # smallest pipe opening in pixels
        gap_max: int = 280, # largest pipe opening in pixels
        speed_period: int = 300, # steps per full speed oscillation cycle
        gap_period: int = 450, # steps per full gap size oscillation cycle
        centre_period: int = 360, # steps per full gap centre oscillation cycle
        render_mode: Optional[str] = None, # set to "human" to open a window
        seed: Optional[int] = None,
    ):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.speed_period = speed_period
        self.gap_period = gap_period
        self.centre_period = centre_period
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # pygame display objects, initialised lazily on first render
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None

        self.reset()

    # reset the environment to a fresh episode and return the first observation
    def reset(self) -> np.ndarray:
        self.score = 0
        self.steps = 0
        self.done = False

        # place the bird in the middle of the screen with no vertical velocity
        self.bird_y = float(SCREEN_H // 2)
        self.bird_vel = 0.0

        # randomise the starting phase of each oscillation so episodes vary
        self._speed_phase = float(self.rng.uniform(0, 2 * math.pi))
        self._gap_phase = float(self.rng.uniform(0, 2 * math.pi))

        self.current_speed = self._calc_speed()
        self.current_gap_size = self._calc_gap_size()

        # start with two pipes already positioned ahead of the bird
        self.pipes: List[Dict] = []
        self._spawn_pipe(SCREEN_W + 60)
        self._spawn_pipe(SCREEN_W + 60 + PIPE_SPACING)

        return self._obs()

    # advance the simulation by one step and return the new observation, reward, done flag, and info
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode ended - call reset().")

        self.steps += 1

        # apply flap velocity if the agent chose to flap, then apply gravity
        if action == 1:
            self.bird_vel = BASE_FLAP_VEL
        self.bird_vel += BASE_GRAVITY
        self.bird_y += self.bird_vel

        # advance both oscillation phases by the appropriate amount for this step
        self._speed_phase += (2 * math.pi) / self.speed_period
        phase_step = (2 * math.pi) / self.gap_period * self.current_speed
        self._gap_phase += phase_step
        self.current_speed = self._calc_speed()
        self.current_gap_size = self._calc_gap_size()

        # update each pipe's gap size and move its centre based on the oscillation
        centre_step = (2 * math.pi) / self.centre_period * self.current_speed
        for p in self.pipes:
            p["centre_phase"] += centre_step
            gap = self.current_gap_size
            # keep the gap centre far enough from the top and bottom edges
            margin = gap // 2 + 40
            base = int(np.clip(p["gap_centre_base"], margin, SCREEN_H - margin))
            p["gap_centre_base"] = base
            # add a sine wave offset on top of the base position
            offset = int(p["centre_amp"] * math.sin(p["centre_phase"]))
            gc = int(np.clip(base + offset, margin, SCREEN_H - margin))
            p["gap_centre"] = gc
            p["gap"] = gap

        # scroll all pipes to the left by the current speed
        pipe_scroll = BASE_SPEED * self.current_speed
        for p in self.pipes:
            p["x"] -= pipe_scroll

        # remove the leftmost pipe once it leaves the screen and spawn a new one at the right
        if self.pipes[0]["x"] + PIPE_W < 0:
            self.pipes.pop(0)
            self._spawn_pipe(self.pipes[-1]["x"] + PIPE_SPACING)

        # check if the bird hit the ceiling or the floor
        if self.bird_y - BIRD_RADIUS < 0 or self.bird_y + BIRD_RADIUS > SCREEN_H:
            self.done = True
            return self._obs(), -5.0, True, {"score": self.score}

        # check for pipe collisions and count any pipes the bird successfully passed
        reward = 0.1
        for p in self.pipes:
            px, gc, g = p["x"], p["gap_centre"], p["gap"]
            top_edge = gc - g // 2
            bot_edge = gc + g // 2

            # the bird overlaps horizontally with this pipe, so check vertical collision
            if px < BIRD_X + BIRD_RADIUS and px + PIPE_W > BIRD_X - BIRD_RADIUS:
                if self.bird_y - BIRD_RADIUS < top_edge or self.bird_y + BIRD_RADIUS > bot_edge:
                    self.done = True
                    return self._obs(), -5.0, True, {"score": self.score}

            # award a point if the bird just cleared this pipe
            if not p["passed"] and px + PIPE_W < BIRD_X - BIRD_RADIUS:
                p["passed"] = True
                self.score += 1
                reward = 1.0

        # small shaping reward for flying close to the centre of the next gap
        next_p = self._next_pipe()
        dist_to_gap = abs(self.bird_y - next_p["gap_centre"]) / SCREEN_H
        reward += 0.1 * (1.0 - dist_to_gap)

        if self.render_mode == "human":
            self._render()

        return self._obs(), reward, False, {"score": self.score}

    # shut down the pygame window if one is open
    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # compute the current speed multiplier using a sine wave over training steps
    def _calc_speed(self) -> float:
        t = (math.sin(self._speed_phase) + 1) / 2
        base = self.speed_min + t * (self.speed_max - self.speed_min)
        # add small random noise to make the speed less predictable
        noise = self.rng.normal(0, 0.05)
        return float(np.clip(base + noise, self.speed_min, self.speed_max))

    # compute the current gap size in pixels using a sine wave over training steps
    def _calc_gap_size(self) -> int:
        t = (math.sin(self._gap_phase) + 1) / 2
        return int(self.gap_min + t * (self.gap_max - self.gap_min))

    # add a new pipe at the given x position with a random centre and oscillation settings
    def _spawn_pipe(self, x: float):
        gap = self.current_gap_size
        margin = gap // 2 + 40
        # pick a random vertical centre that keeps the gap away from the edges
        gc = int(self.rng.uniform(margin, SCREEN_H - margin))
        centre_phase = float(self.rng.uniform(0, 2 * math.pi))
        centre_amp = int(self.rng.uniform(40, 100))
        self.pipes.append({
            "x": x,
            "gap_centre": gc,
            "gap_centre_base": gc,
            "centre_phase": centre_phase,
            "centre_amp": centre_amp,
            "gap": gap,
            "passed": False,
        })

    # return the first pipe that the bird has not yet fully passed
    def _next_pipe(self) -> Dict:
        for p in self.pipes:
            if p["x"] + PIPE_W > BIRD_X - BIRD_RADIUS:
                return p
        return self.pipes[-1]

    # build the 7-float observation vector from the current game state
    def _obs(self) -> np.ndarray:
        p = self._next_pipe()
        # normalise speed to [0, 1] within the configured speed range
        speed_norm = (self.current_speed - self.speed_min) / max(
            self.speed_max - self.speed_min, 1e-6
        )
        # normalise gap phase to [0, 1] using the sine of the current phase
        gap_phase_norm = (math.sin(self._gap_phase) + 1) / 2
        return np.array(
            [
                self.bird_y / SCREEN_H,
                self.bird_vel / 20.0,
                (p["x"] - BIRD_X) / SCREEN_W,
                p["gap_centre"] / SCREEN_H,
                p["gap"] / SCREEN_H,
                speed_norm,
                gap_phase_norm,
            ],
            dtype=np.float32,
        )

    # draw the current game frame to the pygame window
    def _render(self):
        # initialise pygame and the display on the very first render call
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Dynamic Flappy Bird - RL")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 36)
            self._small_font = pygame.font.Font(None, 24)

        # allow the user to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        # sky blue background
        self._screen.fill((135, 206, 235))

        # pipes shift from green to red as the gap narrows
        gap_frac = (self.current_gap_size - self.gap_min) / max(
            self.gap_max - self.gap_min, 1
        )
        r = int(200 * (1 - gap_frac))
        g = int(160 * gap_frac + 40)
        pipe_colour = (r, g, 0)
        cap_colour = tuple(max(0, c - 40) for c in pipe_colour)

        # draw the top and bottom sections of each pipe plus the cap
        for p in self.pipes:
            gc, gap, px = p["gap_centre"], p["gap"], int(p["x"])
            top = gc - gap // 2
            bot = gc + gap // 2
            pygame.draw.rect(self._screen, pipe_colour, (px, 0, PIPE_W, top))
            pygame.draw.rect(self._screen, pipe_colour, (px, bot, PIPE_W, SCREEN_H - bot))
            pygame.draw.rect(self._screen, cap_colour, (px - 3, top - 12, PIPE_W + 6, 12))
            pygame.draw.rect(self._screen, cap_colour, (px - 3, bot, PIPE_W + 6, 12))

        # draw the bird as a yellow circle
        pygame.draw.circle(
            self._screen, (255, 215, 0), (BIRD_X, int(self.bird_y)), BIRD_RADIUS
        )

        # display the current score in the top left corner
        score_txt = self._font.render(f"Score: {self.score}", True, (0, 0, 0))
        self._screen.blit(score_txt, (10, 10))

        # draw speed and gap size indicator bars below the score
        speed_frac = (self.current_speed - self.speed_min) / max(
            self.speed_max - self.speed_min, 1e-6
        )
        self._draw_bar(10, 50, 120, 12, speed_frac, (220, 80, 80), "Speed")
        gap_frac2 = (self.current_gap_size - self.gap_min) / max(
            self.gap_max - self.gap_min, 1
        )
        self._draw_bar(10, 72, 120, 12, gap_frac2, (80, 180, 80), "Gap")

        # show numeric labels next to each bar
        spd_lbl = self._small_font.render(
            f"{self.current_speed:.2f}x  ({BASE_SPEED * self.current_speed:.1f} px/f)", True, (30, 30, 30)
        )
        gap_lbl = self._small_font.render(f"{self.current_gap_size} px", True, (30, 30, 30))
        self._screen.blit(spd_lbl, (140, 48))
        self._screen.blit(gap_lbl, (140, 70))

        pygame.display.flip()
        self._clock.tick(FPS)

    # draw a labelled progress bar at the given screen position
    def _draw_bar(self, x, y, w, h, frac, colour, label):
        lbl = self._small_font.render(label, True, (30, 30, 30))
        self._screen.blit(lbl, (x, y - 14))
        # grey background track
        pygame.draw.rect(self._screen, (180, 180, 180), (x, y, w, h))
        # coloured fill proportional to frac
        pygame.draw.rect(self._screen, colour, (x, y, int(w * frac), h))
        # dark border
        pygame.draw.rect(self._screen, (60, 60, 60), (x, y, w, h), 1)


# fixed environment that disables all oscillation so speed and gap stay constant
class FixedFlappyEnv(DynamicFlappyEnv):

    # create a static environment with a single fixed speed and gap size
    def __init__(
        self,
        speed: float = 1.0,
        gap_size: int = 200,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        # pass identical min and max values so the oscillation range collapses to zero
        super().__init__(
            speed_min = speed,
            speed_max = speed,
            gap_min = gap_size,
            gap_max = gap_size,
            speed_period = 1, # period is irrelevant when the range is zero
            gap_period = 1,
            render_mode = render_mode,
            seed = seed,
        )

    # always return the fixed speed with no noise
    def _calc_speed(self) -> float:
        return self.speed_min

    # always return the fixed gap size
    def _calc_gap_size(self) -> int:
        return self.gap_min

    # spawn a pipe with a fixed centre that never moves
    def _spawn_pipe(self, x: float):
        gap = self.current_gap_size
        margin = gap // 2 + 40
        gc = int(self.rng.uniform(margin, SCREEN_H - margin))
        self.pipes.append({
            "x": x,
            "gap_centre": gc,
            "gap_centre_base": gc,
            "centre_phase": 0.0,
            "centre_amp": 0, # zero amplitude means the centre never oscillates
            "gap": gap,
            "passed": False,
        })