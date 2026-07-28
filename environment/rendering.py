"""
Pygame renderer for ClinicRestockEnv.

Draws each drug as a vertical stock bar (green/yellow/red zones for
healthy/low/critical stock), plus a header showing the current week,
season, and running stockout count. Supports render_mode="human" (opens
a window) and "rgb_array" (returns a numpy frame, useful for saving video
without a display -- e.g. inside the demo recording script).
"""

import numpy as np
import pygame

WIDTH, HEIGHT = 800, 480
BAR_WIDTH = 120
BAR_MAX_HEIGHT = 300
MARGIN_TOP = 120

COLOR_BG = (18, 18, 24)
COLOR_TEXT = (235, 235, 235)
COLOR_GOOD = (70, 190, 110)
COLOR_LOW = (230, 190, 60)
COLOR_CRITICAL = (220, 70, 70)
COLOR_BAR_BG = (50, 50, 60)


class ClinicRenderer:
    def __init__(self, drug_names, max_capacity, render_mode="human"):
        pygame.init()
        self.drug_names = drug_names
        self.max_capacity = max_capacity
        self.render_mode = render_mode
        self.font = pygame.font.SysFont("arial", 18)
        self.font_big = pygame.font.SysFont("arial", 26, bold=True)

        if render_mode == "human":
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Rural Clinic Medicine Restocking Agent")
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()

    def _stock_color(self, frac):
        if frac > 0.5:
            return COLOR_GOOD
        elif frac > 0.2:
            return COLOR_LOW
        return COLOR_CRITICAL

    def render(self, env):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        self.screen.fill(COLOR_BG)

        title = self.font_big.render("Rural Clinic Medicine Restocking Agent", True, COLOR_TEXT)
        self.screen.blit(title, (20, 15))

        season_txt = "MALARIA SEASON" if env.week and (
            9 <= (env.week % 52) <= 22 or 40 <= (env.week % 52) <= 48
        ) else "off-season"
        info_txt = (
            f"Week {env.week}/{env.episode_length}   |   {season_txt}   |   "
            f"Cumulative unmet demand: {env.total_stockouts:.0f} units"
        )
        info_surf = self.font.render(info_txt, True, COLOR_TEXT)
        self.screen.blit(info_surf, (20, 55))

        n = len(self.drug_names)
        gap = WIDTH // (n + 1)
        for i, name in enumerate(self.drug_names):
            x = gap * (i + 1) - BAR_WIDTH // 2
            frac = float(np.clip(env.stock[i] / self.max_capacity[i], 0, 1))
            bar_h = int(BAR_MAX_HEIGHT * frac)
            # background track
            pygame.draw.rect(
                self.screen, COLOR_BAR_BG,
                (x, MARGIN_TOP, BAR_WIDTH, BAR_MAX_HEIGHT), border_radius=6
            )
            # filled stock
            pygame.draw.rect(
                self.screen, self._stock_color(frac),
                (x, MARGIN_TOP + (BAR_MAX_HEIGHT - bar_h), BAR_WIDTH, bar_h), border_radius=6
            )
            label = self.font.render(name, True, COLOR_TEXT)
            self.screen.blit(label, (x, MARGIN_TOP + BAR_MAX_HEIGHT + 10))
            stock_val = self.font.render(
                f"{env.stock[i]:.0f}/{int(self.max_capacity[i])}", True, COLOR_TEXT
            )
            self.screen.blit(stock_val, (x, MARGIN_TOP - 22))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(4)
            return None
        else:
            frame = pygame.surfarray.array3d(self.screen)
            return np.transpose(frame, (1, 0, 2))

    def close(self):
        pygame.quit()
