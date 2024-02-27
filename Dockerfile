# Use the Python 3.9 base image
FROM python:3.9

# Set the working directory inside the container to /code
WORKDIR /code

# Copy the requirements.txt file from the host to the container's /code directory
COPY ./requirements.txt /code/requirements.txt

# Install Python packages listed in requirements.txt without caching and upgrade if necessary
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a user named "user" with a user ID of 1000 inside the container for security
RUN useradd -m -u 1000 user

# Set the user "user" as the user for subsequent commands
USER user

# Define environment variables HOME and PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change the working directory to the user's home directory /home/user/app
WORKDIR $HOME/app

# Copy the contents of the current directory into the container's /home/user/app directory,
# ensuring that files are owned by the "user"
COPY --chown=user . $HOME/app

# Specify the default command to run the FastAPI application with UVicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]