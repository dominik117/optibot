# OptiBot

OptiBot enhances chatbot interactions using NLP and machine learning to identify and address common response issues, boosting accuracy and user satisfaction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher. Python 3.12 is recommended.
- pipx (Python package installer)

### Installation

#### Poetry for Dependency Management

This project uses Poetry for dependency management. If you don't have Poetry installed, it is recommended installing it using pipx. Refer to the [pipx installation guide](https://pypa.github.io/pipx/installation/) if pipx is not installed on your system.

#### Project Setup

1. **Clone the Repository**

   Begin by cloning the repository to your local machine:

   ```bash
   git clone git@github.com:dominik117/optibot.git
   cd optibot
   ```

2. **Install Poetry with pipx**

   If pipx is installed but Poetry is not yet, install Poetry using the following command:

   ```bash
   pipx install poetry
   ```

3. **Initialize the Project with Poetry**

   Set up the project using Poetry:

   ```bash
   poetry install
   ```

### Running the Application

To run the OptiBot application, use one of the following commands:

To run in a Streamlit App:

```bash
poetry run optibot
```

To run in the terminal (Currently under development):

```bash
poetry run optibot-terminal
```

Follow the on-app instructions to interact with the application.


## License

See the `LICENSE` file for details.

## Background

This project began as a final project for the Generative AI class at the University of Applied Sciences in Lucerne, imparted by [Dr. Marcel Blattner](https://www.linkedin.com/in/marcelblattner/). Thanks to Dr. Blattner for his guidance and insights throughout the development of Optibot.

## References

Refer to the [References](./references/) folder for more on this project.



