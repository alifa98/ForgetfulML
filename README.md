# ForgetfulML

> ⚠️ Warning
> This Library is under development.

**ForgetfulML** is a machine unlearning framework designed to enable AI models to selectively forget previously learned data. This functionality is crucial for ensuring compliance with privacy regulations, improving model adaptability, and maintaining ethical standards in machine learning. ForgetfulML simplifies the process of unlearning, offering a seamless integration with existing ML workflows.

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

Here's how you can quickly start using **ForgetfulML** in your project:

```python
# comming soon
```

## Use Cases

- **Data Privacy**: Comply with privacy laws by unlearning user data upon request.
- **Model Updates**: Keep your models up to date by removing outdated or irrelevant data.
- **Data Debugging**: Correct model misbehavior by unlearning incorrect or mislabeled training data.

## Documentation

⚠️ Under development
[Wiki Website](https://github.com/alifa98/ForgetfulML/wiki)

## Contributing

We welcome contributions! If you'd like to help improve ForgetfulML, feel free to submit pull requests or open issues.

## Support

For questions or issues, feel free to reach out by opening a GitHub issue or contact us at <unlearning@faraji.info>

## Development: Build

Install build dependencies:

```bash
pip install build
```

Build the package:

```bash
python -m build
```

## Development: Run Tests

Install test dependencies (tests/requirements.txt):

```bash
pip install -r tests/requirements.txt
```

Run the tests:

```bash
pytest
```
