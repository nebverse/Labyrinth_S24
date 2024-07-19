# Labyrinth_S24

# Raspberry Pi Setup Guide

This guide provides step-by-step instructions for setting up a Raspberry Pi with Raspberry Pi OS and configuring it for use with Dynamixel MX-28AT motors. It also includes information on setting up a virtual environment, installing necessary software packages, and connecting peripherals.

## Requirements

- **SD Card**: 32 GB (recommended)
- **Raspberry Pi**: Any model with necessary peripherals
- **Power Supply**: 7.5V via pixl card attached to 1-8 pins
- **Internet Connection**: Ethernet cable
- **Monitor**: HDMI cable
- **Keyboard and Mouse**: USB port
- **Dynamixel Motor**: MX-28AT
- **Camera**: External USB webcam

## Setup Steps

### 1. Install Raspberry Pi OS

1. Download and install the Raspberry Pi Imager from [Raspberry Pi Software](https://www.raspberrypi.com/software/).
2. Use the Raspberry Pi Imager to install Raspberry Pi OS on the SD card (32 GB recommended).
3. Follow the steps mentioned in the Raspberry Pi Imager.

### 2. Initial Setup

1. Insert the SD card with Raspberry Pi OS into the Raspberry Pi.
2. Connect the power supply via pixl card attached to 1-8 pins (7.5V).
3. Connect the Ethernet cable for internet access.
4. Connect the HDMI cable to the monitor.
5. Connect the keyboard and mouse to the USB ports.

### 3. User Configuration

- **Username**: `cyber`
- **Password**: `runner`

### 4. Dynamixel MX-28AT Motor Setup

1. Follow the instructions in the [Poppy Project Documentation](https://docs.poppy-project.org/en/assembly-guides/poppy-torso/addressing_dynamixel#connect-a-single-motor-to-configure-it) to connect the Dynamixel MX-28AT motor.
2. If you need to reconfigure the motor settings, use the Dynamixel Wizard.
3. If you encounter a permission error with `/dev/ttyACM0`, run the following command:
    ```bash
    sudo chmod 666 /dev/ttyACM0
    ```
    Refer to this [Stack Overflow post](https://stackoverflow.com/questions/27858041/oserror-errno-13-permission-denied-dev-ttyacm0-using-pyserial-from-pyth) for more details.

### 5. Software Setup

1. **Setup Virtual Environment**:
   Follow the instructions in this [Stack Overflow post](https://stackoverflow.com/questions/75602063/pip-install-r-requirements-txt-is-failing-this-environment-is-externally-mana/75696359#75696359) to set up a virtual environment inside Raspberry Pi.
2. **Install Pypot Package**:
   Use the instructions from the [Poppy Project Documentation](https://docs.poppy-project.org/en/software-libraries/pypot) to install the package necessary for accessing motors through Python.

### 6. Camera Setup

1. Use an external USB webcam.
2. Follow the setup guide in this [Medium article](https://medium.com/@robotamateur123/use-a-usb-camera-with-raspberry-pi-for-beginners-5f0ed8e98400).

### 7. Model-based Reinforcement Learning

For implementing model-based reinforcement learning, refer to this [arXiv paper](https://arxiv.org/pdf/2312.09906).

## Notes

- Ensure all connections are secure and double-check the power supply specifications to avoid any damage to the Raspberry Pi and connected peripherals.
- Regularly update the Raspberry Pi OS and installed packages to maintain security and functionality.

Feel free to reach out if you have any questions or encounter issues during the setup process. Happy tinkering!
