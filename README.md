# ForgetfulML

**ForgetfulML** is a cutting-edge machine unlearning framework designed to enable AI models to selectively forget previously learned data. This functionality is crucial for ensuring compliance with privacy regulations, improving model adaptability, and maintaining ethical standards in machine learning. ForgetfulML simplifies the process of unlearning, offering a seamless integration with existing ML workflows.

## Key Features

- **Selective Unlearning**: Remove specific data or knowledge from AI models while maintainig the performance.
- **Compliance & Privacy**: Comply with privacy regulations like GDPR by unlearning personal or sensitive information.
- **Easy Integration**: Compatible with popular machine learning frameworks.
- **Adaptability**: Helps AI models evolve by erasing outdated or irrelevant information.
- **Evaluation**: Provides Tools to evaluate your unlearning methods.
  
## Installation

Install ForgetfulML using pip:

```bash
pip install unlearning
```

## Quick Start

> ⚠️ Warning
> This Library is under development.

Here's how you can quickly start using **ForgetfulML** in your project:

```python
# Import the Unlearning module
from unlearning import ClassForgetter

# Initialize the Forgetter
forgetter = ClassForgetter()

# Load your pre-trained machine learning model
model = load_your_model()  # Replace with your model loading function

# Define the data you want the model to forget
data_to_forget = get_data_to_forget()  # Replace with your data extraction

# Unlearn the specified data
forgetter.unlearn(model, data_to_forget)

# Continue using your updated model
save_model(model)  # Replace with your model saving function
```

## Use Cases

- **Data Privacy**: Comply with privacy laws by unlearning user data upon request.
- **Model Updates**: Keep your models up to date by removing outdated or irrelevant data.
- **Data Debugging**: Correct model misbehavior by unlearning incorrect or mislabeled training data.

## Documentation

⚠️ Under development

## Contributing

We welcome contributions! If you'd like to help improve ForgetfulML, feel free to submit pull requests or open issues.

## Support

For questions or issues, feel free to reach out by opening a GitHub issue or contact us at unlearning@faraji.info


# Development

## Build

Install build dependencies:

```bash
pip install build
```

Build the package:

```bash
python -m build
```

## Test

Install test dependencies (tests/requirements.txt):

```bash
pip install -r tests/requirements.txt
```

Run the tests:

```bash
pytest
```
