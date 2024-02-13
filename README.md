# Calc-X and Calcformers

You can access the datasets and the trained models on HuggingFace - <https://huggingface.co/anonym-repos>

This repo contains dataset builders, training scripts, and inference wrappers for training and using Calcformers, models capable of using a calculator during inference. It supports supervised and preference optimization training. The training and evaluation scripts are in `examples` directory.

## Create the environment

First, clone the repo. Then run:

```shell
conda create -n gadgets python=3.10 && conda activate gadgets
pip install poetry
poetry install
```

This installs all dependencies in exact same versions used by the authors of the repo.
In case you encounter any issues on your hardware (e.g., with CUDA version, platform, etc.),
you can resolve the dependencies yourself:

```shell
# with plain pip:
pip install -e .[dev]
# OR with poetry:
poetry lock && poetry install
```

## Usage

We wrap the `generate()` method to be able to utilize the
given set of gadgets during the generation.
You will need to wrap the model of your choice and
make sure that the tokenizer is able to encode the instruction
HTML tags used in calling the gadget calls.

Using our pre-trained models (with the tokenizer resolved),
you can use the model using a calculator gadget as follows.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

from gadgets.model import gadget_assisted_model
from gadgets.gadget import Calculator

GadgetAssistedT5 = gadget_assisted_model(T5ForConditionalGeneration)

model = GadgetAssistedT5.from_pretrained("anonym-repos/calcformer-flan-xl")
tokenizer = T5Tokenizer.from_pretrained("anonym-repos/calcformer-flan-xl")

model.prepare_for_generate(tokenizer,
                           enabled_gadgets=[Calculator()],
                           default_max_tokens=512)
query = """
    The profit from a business transaction is shared among 2 business partners,
    Mike and Johnson in the ratio 2:5 respectively.
    If Johnson got $2500, how much will Mike have
    after spending some of his share on a shirt that costs $200?
"""

inputs = tokenizer(query, return_tensors="pt")
output_ids = model.generate(**inputs)
tokenizer.decode(output_ids[0], spaces_between_special_tokens=False)

# This returns:
# 'According to the ratio, Mike got 2/5*$2500 = $<gadget id="calculator">2/5*2500</gadget><output>1_000</output> 1000
#  Mike will have $1000-$200 = $<gadget id="calculator">1000-200</gadget><output>800</output> 800 after buying a shirt.
#  Final result is<result>800</result></s>'
```

If you use a decoder-only model, pass the `architecture` parameter into `model.generate` as follows:

```python
output_ids = model.generate(**inputs, architecture='decoder-only')
```
