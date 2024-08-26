# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Models training logic.
"""
import logging
import time
from typing import Iterator, Optional
import gin
import numpy as np
import torch as t
from torch import optim
from common.torch.losses import smape_2_loss, mape_loss, mase_loss
from common.torch.snapshots import SnapshotManager
from common.torch.ops import default_device, to_tensor

# Setup logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()  # You can add a FileHandler here to save logs to a file
                    ])

@gin.configurable
def trainer(snapshot_manager: SnapshotManager,
            model: t.nn.Module,
            training_set: Iterator,
            timeseries_frequency: int,
            loss_name: str,
            iterations: int,
            learning_rate: float = 0.001):

    model = model.to(default_device())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_loss_fn = __loss_fn(loss_name)

    lr_decay_step = iterations // 3
    if lr_decay_step == 0:
        lr_decay_step = 1

    iteration = snapshot_manager.restore(model, optimizer)

    #
    # Training Loop
    #
    snapshot_manager.enable_time_tracking()
    logging.info(f'Starting training for {iterations} iterations...')
    for i in range(iteration + 1, iterations + 1):
        start_time = time.time()
        model.train()

        # Fetch the next batch of data
        batch = next(training_set)
        
        # Handle both masked and non-masked data
        if len(batch) == 2:
            x, y = map(to_tensor, batch)
            x_mask, y_mask = None, None  # No masks provided
        elif len(batch) == 4:
            x, x_mask, y, y_mask = map(to_tensor, batch)
        else:
            raise ValueError(f"Expected 2 or 4 elements in batch, but got {len(batch)}")

        # Use default masks (ones) if none are provided
        if x_mask is None:
            x_mask = t.ones_like(x)
        if y_mask is None:
            y_mask = t.ones_like(y)

        optimizer.zero_grad()
        forecast = model(x, x_mask)
        training_loss = training_loss_fn(x, timeseries_frequency, forecast, y, y_mask)

        if np.isnan(float(training_loss)):
            logging.error(f"Training stopped due to NaN loss at iteration {i}.")
            print(f"Warning: NaN loss detected at iteration {i}. Stopping training.")
            break

        training_loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)

        # Log progress
        time_taken = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f'Iteration {i}/{iterations}, Loss: {training_loss:.6f}, LR: {current_lr:.6f}, Time: {time_taken:.2f}s')

        # Print progress and performance
        if i % 10 == 0 or i == iterations:  # Print every 10 iterations and at the end
            print(f"Iteration {i}/{iterations}, Training Loss: {float(training_loss):.4f}")

        snapshot_manager.register(iteration=i,
                                  training_loss=float(training_loss),
                                  validation_loss=np.nan, model=model,
                                  optimizer=optimizer)

    logging.info('Training completed.')
    print("Training complete.")
    return model

def __loss_fn(loss_name: str):
    def loss(x, freq, forecast, target, target_mask):
        if loss_name == 'MAPE':
            return mape_loss(forecast, target, target_mask if target_mask is not None else t.ones_like(target))
        elif loss_name == 'MASE':
            return mase_loss(x, freq, forecast, target, target_mask if target_mask is not None else t.ones_like(target))
        elif loss_name == 'SMAPE':
            return smape_2_loss(forecast, target, target_mask if target_mask is not None else t.ones_like(target))
        else:
            raise Exception(f'Unknown loss function: {loss_name}')
    return loss