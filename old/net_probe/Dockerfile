# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY net-probe-server.py /app


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir iperf3
RUN apt-get update && apt-get install -y iperf3

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run net_probe/net-probe-server.py when the container launches
CMD ["python", "net-probe-server.py"]