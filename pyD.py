import usb.core
import usb.util
import usb.backend.libusb1 as libusb1
import os
import serial.tools.list_ports

# Initialize macOS backend path
backend = None
mac_brew_path = "/opt/homebrew/lib/libusb-1.0.dylib"
if os.path.exists(mac_brew_path):
    backend = libusb1.get_backend(find_library=lambda name: mac_brew_path)

# TARGET_PORT: Set this to the port number you want to target
TARGET_PORT = 1 

# Find all devices
devices = usb.core.find(find_all=True, backend=backend)

for dev in devices:
    if dev.port_number == TARGET_PORT:
        
        # 1. Extract Unique Hardware Serial Number
        try:
            unique_id = dev.serial_number
        except Exception:
            unique_id = None

        # 2. Extract Device Names
        try:
            manufacturer = usb.util.get_string(dev, dev.iManufacturer)
            product = usb.util.get_string(dev, dev.iProduct)
            device_name = f"{manufacturer} {product}"
        except Exception:
            device_name = f"Unknown Device [VID: {dev.idVendor:04X}, PID: {dev.idProduct:04X}]"

        # 3. Match with Serial Port (COM / tty)
        assigned_port = "No Serial Port Found"
        if unique_id:
            # Look through all system serial ports for a matching hardware ID
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if p.serial_number == unique_id:
                    assigned_port = p.device
                    break

        print(f"--- Device Found on Port {TARGET_PORT} ---")
        print(f"Name: {device_name}")
        print(f"Unique Serial ID: {unique_id or 'No Serial Number Available'}")
        print(f"System Serial Port: {assigned_port}")  # This is the new line
        print(f"Full Hub Path: {dev.port_numbers}")
        break
else:
    print(f"No USB device found active on port {TARGET_PORT}")