import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "disabled"

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from bergson.data import load_gradients
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)
from bergson.utils.utils import assert_type


class TestGradientCollectorCallback:
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
        return AutoModelForCausalLM.from_config(config)

    @pytest.fixture
    def dataset(self):
        """Create a small test dataset."""
        data = {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            "labels": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
        }
        return Dataset.from_dict(data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_gpu_order_tracking(self, tmp_path, model, dataset):
        """Test that every step has an associated order record in single-GPU mode."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify order records were created
        assert callback.order is not None
        assert len(callback.order) > 0

        # Check that every step has associated order records
        steps_with_records = set()
        for record in callback.order:
            steps_with_records.add(record["global_step"])

        # Get the actual number of training steps
        expected_steps = len(dataset) // training_args.per_device_train_batch_size
        if len(dataset) % training_args.per_device_train_batch_size != 0:
            expected_steps += 1

        # Verify we have records for all expected steps
        assert len(steps_with_records) == expected_steps
        # Expected steps are 1-indexed
        assert steps_with_records == set(range(1, expected_steps + 1))

        # Verify each record has required fields
        for record in callback.order:
            assert "_idx" in record
            assert "global_step" in record
            assert "epoch" in record
            assert isinstance(record["_idx"], int)
            assert isinstance(record["global_step"], int)
            assert isinstance(record["epoch"], int)

        # Verify indices are within valid range
        for record in callback.order:
            assert 0 <= record["_idx"] < len(dataset)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_order_tracking_disabled(self, tmp_path, model, dataset):
        """Test that no order records are created when tracking is disabled."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients", use_optimizer_state=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify no order records were created
        assert callback.order is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_order_save_and_load(self, tmp_path, model, dataset):
        """Test that order records are properly saved and can be loaded."""
        # Train the model with the callback
        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify order records were created
        assert callback.order is not None
        assert len(callback.order) > 0

        # Check that order file was saved
        order_file = tmp_path / "gradients" / "order.hf"
        assert order_file.exists()

        # Load and verify the saved order
        saved_order = Dataset.load_from_disk(str(order_file))
        assert len(saved_order) == len(callback.order)

        # Verify the saved order matches the in-memory order
        for i, record in enumerate(saved_order):
            record = assert_type(dict, record)
            assert record["_idx"] == callback.order[i]["_idx"]
            assert record["global_step"] == callback.order[i]["global_step"]
            assert record["epoch"] == callback.order[i]["epoch"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sft_trainer(self, tmp_path, model, dataset):
        """Test that gradient and order files are created and
        can be loaded after training with SFTTrainer."""
        # Set up and train the model with SFT
        sft_config = SFTConfig(
            output_dir=str(tmp_path / "sft_output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
        )

        callback = GradientCollectorCallback(
            path=tmp_path / "gradients",
            track_order=True,
            use_optimizer_state=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[callback],
        )
        trainer = prepare_for_gradient_collection(trainer)
        trainer.train()

        # Verify training order was tracked
        assert callback.order is not None
        assert len(callback.order) > 0

        # Verify gradient files were created
        gradient_dir = tmp_path / "gradients"
        train_gradient_dir = gradient_dir / "train" / "epoch_0"

        assert gradient_dir.exists()
        assert train_gradient_dir.exists()
        assert (train_gradient_dir / "gradients.bin").exists()
        assert (train_gradient_dir / "info.json").exists()
        assert (gradient_dir / "order.hf").exists()

        # Test loading the gradient data directly
        gradients = load_gradients(train_gradient_dir)
        assert len(gradients) > 0

        # Verify order data was saved and can be loaded
        order_file = tmp_path / "gradients" / "order.hf"
        assert order_file.exists()

        saved_order = Dataset.load_from_disk(str(order_file))
        assert len(saved_order) > 0
        assert all(key in saved_order[0] for key in ["_idx", "global_step", "epoch"])
