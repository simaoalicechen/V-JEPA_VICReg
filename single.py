'''
have not considered noise 
'''
'''
based on https://github.com/vladisai/JEPA_SSL_NeurIPS_2022/blob/main/data/single.py
'''
from typing import NamedTuple, Any, Optional
import math

import torch
import torchvision
from torchvision import transforms
import numpy as np

class Sample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]

class ContinuousMotionDataset:
    def __init__(
        self,
        size,
        batch_size = 32,
        # changed to 20 steps or 20 frames in total 
        n_steps=20,
        concentration = 0.2,
        max_step = 4.0,
        std = 1.3,
        img_size = 28,
        normalize = False,
        device = torch.device("cpu"),
        train = True,
    ):
        self.size = size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.concentration = concentration
        self.std = std
        self.max_step = max_step
        self.img_size = img_size
        self.device = device
        self.normalize = False
        self.padding = 2 * self.std
        if normalize:
            self.n_mean, self.n_std = self.estimate_mean_std()
            self.normalize = True

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.generate_multistep_sample()

    def __iter__(self):
        for _ in range(self.size):
            yield self.generate_multistep_sample()

    def render_location(self, locations: torch.Tensor):
        x = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        c = c.view(*([1] * (len(locations.shape) - 1)), *c.shape).repeat(
            *locations.shape[:-1], *([1] * len(c.shape))
        )  # repeat the number of times required for locations.
        locations = locations.unsqueeze(-2).unsqueeze(
            -2
        )  # add dims for compatibility with c
        img = torch.exp(
            -(c - locations).norm(dim=-1).pow(2) / (2 * self.std * self.std)
        ) / (2 * math.pi * self.std * self.std)
        # img_rgb = img.unsqueeze(-3).repeat(..., 3, 1, 1) 
        return img

    def generate_state(self):
        # We leave 2 * self.std margin when generating state, and don't let the
        # dot approach the border.
        effective_range = (self.img_size - 1) - 2 * self.padding
        location = (
            torch.rand(size=(self.batch_size, 2), device=self.device) * effective_range
            + self.padding
        )
        return location

    def get_images_for_overlay(self):
        try:
            batch = next(self.loader_it)
        except StopIteration:
            self.loader_it = iter(self.loader)
            batch = next(self.loader_it)

        batch = batch[0][:, :, :28, :28]  # crop from 32 by 32 to 28 by 28
        # batch /= batch.max(dim=0).values asdf
        return batch.to(self.device)


    def generate_multistep_sample(
        self,
    ):
        actions = self.generate_actions(self.n_steps)
        start_location = self.generate_state()
        sample = self.generate_transitions(start_location, actions)

        if self.normalize:
            sample = Sample(
                states=(sample.states - self.n_mean) / self.n_std,
                locations=(
                    (sample.locations - self.padding)
                    / (self.img_size - 1 - 2 * self.padding)
                    - 0.5
                )
                / np.sqrt(1 / 12),
                actions=sample.actions / (0.4102 * self.max_step + 1e-7),
            )
        return sample

    def normalize_location(self, location):
        return (
            (location - self.padding) / (self.img_size - 1 - 2 * self.padding) - 0.5
        ) / np.sqrt(1 / 12)

    def generate_transitions(
        self,
        location,
        actions,
    ):
        locations = [location]
        for i in range(actions.shape[1]):
            # print("generate_transitions: ", actions.shape[1])
            next_location = self.generate_transition(locations[-1], actions[:, i])
            locations.append(next_location)

        # Unsqueeze for compatibility with multi-dot dataset
        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        return Sample(states, locations, actions)

    def generate_transition(self, location, action):
        next_location = location + action  # [..., :-1] * action[..., -1]
        # don't let the dot get closer to the border than 2 * self.std
        next_location = torch.clamp(
            next_location, min=self.padding, max=self.img_size - 1 - self.padding
        )
        return next_location

    def generate_actions(self, n_steps: int):
        x = torch.rand(self.batch_size, device=self.device) * 2 * math.pi
        d = torch.distributions.VonMises( 
            x, concentration=self.concentration * torch.ones_like(x)
        )
        a = (
            d.sample((n_steps - 1,)).permute(1, 0).to(self.device)
        )  # put the batch dim first
        step_sizes = (
            torch.rand(self.batch_size, n_steps - 1, 1, device=self.device)
            * self.max_step
        )
        vec = ContinuousMotionDataset.angle_to_vec(a)
        actions = vec * step_sizes
        # print(f"inside generate_actions function, generate_actions output shape: {actions.shape}")
        return actions

    @staticmethod
    def angle_to_vec(a):
        return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)

    @torch.no_grad()
    def estimate_mean_std(self):
        N = 100
        means = []
        stds = []
        for i in range(N):
            imgs = self[i].states.flatten()
            means.append(imgs.mean().item())
            stds.append(imgs.std().item())

        return np.mean(means), np.mean(stds)

    @torch.no_grad()
    def unnormalize_mse(self, mse):
        if not self.normalize:
            return mse
        else:
            return mse * (
                (np.sqrt(1 / 12) * (self.img_size - 1 - 2 * self.padding)) ** 2
            )

    @torch.no_grad()
    def unnormalize_location(self, locations):
        if not self.normalize:
            return locations
        else:
            return ((locations * np.sqrt(1 / 12)) + 0.5) * (
                self.img_size - 1 - 2 * self.padding
            ) + self.padding


class DeterministicMotionDataset(ContinuousMotionDataset):
    def generate_actions(self, n_steps: int):
        x = torch.rand(self.batch_size, 1, device=self.device) * 2 * math.pi
        a = x.repeat(1, n_steps - 1)
        vec = ContinuousMotionDataset.angle_to_vec(a)
        step_sizes = (
            torch.rand(self.batch_size, n_steps - 1, 1, device=self.device)
            * self.max_step
        )
        actions = vec * step_sizes
        return actions
