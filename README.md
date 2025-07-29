# Ontology-based R&D

LLM Output Processing System - Ontology-based Document Information Extraction System

## Overview

This system is designed to extract structured information from PDF files. Using ontology (knowledge systems), it automatically extracts necessary information from financial documents such as purchase confirmations and sale confirmations.

## Folder Structure

- `data`: Contains labeled data and OCR/direct text data in Excel format
- `src`: Source code files
- `template`: Ontology, example, and rule template files
- `prompt`: Prompt output directory
- `ollama_output`: Processing results output directory

## Setup

### 1. Create Configuration File

Create a `config.json` file in the project root. You can copy and use the following sample:

```json
{
  "experiment": {
    "n_exp": 5,
    "temperature": 0.1
  },
  "ontology": {
    "type": "nl",
    "file_expression": "nl_ontology_v1.nl"
  },
  "document": {
    "type": "sale_confirmation",
    "dict_version": "v1"
  },
  "paths": {
    "label_with_data_path": "./data/labels_with_data.xlsx",
    "prompt_dir": "./prompt",
    "output_base_path": "./ollama_output",
    "dict_base_path": "./template/dict",
    "ontology_base_path": "./template/ontology",
    "ontology_example_base_path": "./template/ontology_examples"
  },
  "llm": {
    "ollama": {
      "model": "gemma3n:e4b",
      "url": "http://localhost:11434"
    }
  },
  "ontology_type_config": {
    "jsonld": {
      "folder_name": "jsonld"
    },
    "nl": {
      "folder_name": "nl"
    }
  },
  "doc_type_config": {
    "purchase_confirmation": {
      "abbreviation": "pur",
      "required_headers": ["fund_code", "brand_name", "trade_date", "settlement_date", "base_currency", "settlement_currency", "order_quantity", "unit_price", "gross_amount", "fee", "settlement_amount_base_currency", "settlement_amount_settlement_currency"],
      "num_ids": [1,2,3,4,5,6,7,8,9]
    },
    "sale_confirmation": {
      "abbreviation": "sal",
      "required_headers": ["fund_code", "brand_name", "trade_date", "settlement_date", "base_currency", "settlement_currency", "order_quantity", "unit_price", "gross_amount", "fee", "settlement_amount_base_currency", "settlement_amount_settlement_currency"],
      "num_ids": [1,2,3,4,5,6,7,8,9,10]
    }
  }
}
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

The system uses Ollama for local LLM processing. Make sure Ollama is installed and running on your system.

## Configuration File (config.json) Detailed Description

### experiment Section
Basic settings for experiments

- **n_exp**: Number of experimental runs (default: 5)
  - Specifies how many times to repeat experiments under the same conditions
  - Multiple runs are recommended to improve result reliability

- **temperature**: Controls randomness of LLM output (default: 0.1)
  - Specify in the range 0.0-1.0
  - Low values (0.1) for consistent output, high values (0.8-1.0) for creative output

### ontology Section
Settings related to ontology (knowledge systems)

- **type**: Ontology file format (e.g., "nl", "jsonld")
  - "nl": Natural language format
  - "jsonld": JSON-LD format

- **file_expression**: File name pattern for target ontology files
  - Can be specified as regular expression
  - Example: "nl_ontology_v1.nl"

### document Section
Settings for target documents to process

- **type**: Type of document to process
  - "purchase_confirmation": Purchase confirmation
  - "sale_confirmation": Sale confirmation

- **dict_version**: Version of dictionary file to use (e.g., "v1")

### paths Section
Various file path settings

- **label_with_data_path**: Path to labeled data file
  - Excel file containing correct answer data

- **prompt_dir**: Prompt output directory
  - Saves LLM input and output logs

- **output_base_path**: Base directory for processing results
  - Where analysis result Excel files are saved

- **dict_base_path**: Base directory for dictionary files

- **ontology_base_path**: Base directory for ontology files

- **ontology_example_base_path**: Base directory for ontology examples

### llm Section
Language model related settings

#### ollama Subsection
Settings for Ollama local execution

- **model**: Ollama model name to use (e.g., "gemma3n:e4b")

- **url**: Ollama server URL (e.g., "http://localhost:11434")

### ontology_type_config Section
Settings by ontology type

Specifies folder names for each ontology type (jsonld, nl)

### doc_type_config Section
Detailed settings by document type

For each document type, specify the following:

- **abbreviation**: Abbreviation for document type
  - "pur": Purchase
  - "sal": Sale

- **required_headers**: List of header items that need to be extracted
  - Names of information items to extract from documents

- **num_ids**: List of ID numbers for target files to process
  - Specifies numbers of PDF files to process

## Usage

```bash
python src/llm_output.py
```

## Important Notes for Configuration Changes

1. Specify file paths as relative paths and maintain positional relationships from project root
2. When adding new document types, create corresponding template files as well
3. Ensure Ollama is running and the specified model is available
4. When changing ontology file names, corresponding example and rule files must be updated simultaneously

## Security Considerations

### Protected Files (excluded by .gitignore)
- `config.json`: Contains actual operational settings
- `data/` folder: Contains actual corporate documents and labeled data
- `prompt/` folder: Contains LLM input/output logs
- `ollama_output/` folder: Contains processing results

### Local Processing
This system uses Ollama for local LLM processing, which means:
- No data is sent to external APIs
- All processing is done locally on your machine
- Ensure Ollama service is properly secured if running on a shared system

## Customization Examples

### Adding New Document Types

1. Add new document type to `doc_type_config` section in `config.json`
2. Create corresponding ontology and sample files
3. Create rule conversion files as needed

### Using Different LLM Models

- Update `llm.ollama.model` in config.json to specify different Ollama models
- Ensure the specified model is available in your Ollama installation
- Use `ollama list` command to see available models

## Troubleshooting

### Common Issues

1. **Config file not found**: Ensure `config.json` exists in project root
2. **Ollama connection errors**: Verify Ollama is running and accessible at the configured URL
3. **Model not found**: Check that the specified model is installed with `ollama list`
4. **Path errors**: Check that all paths in config are relative to project root
5. **Missing dependencies**: Run `pip install -r requirements.txt`

### Development Tips

- Start with small document samples for testing
- Use temperature=0.1 for consistent results during development
- Check prompt logs in `prompt/` directory for debugging
- Verify ontology file format matches configuration
