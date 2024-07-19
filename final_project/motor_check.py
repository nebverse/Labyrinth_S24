

# from pypot.dynamixel.io import DxlIO

# with DxlIO('COM7') as dxl_io:
#     motor_IDs = dxl_io.scan()
#     print(motor_IDs)
#     num_motors = len(motor_IDs)
#     print("Found", num_motors, "motors with current angles",  dxl_io.get_present_position(motor_IDs))
#     #dxl_io.set_goal_position(dict(zip(motor_IDs, num_motors*[0])))
#     dxl_io.set_goal_position({0: 0, 1: 0})

# import pypot.dynamixel


# ports = pypot.dynamixel.get_available_ports()
# print('available ports:', ports)

import itertools
import pypot.dynamixel

port = 'COM7'  # Example port, change as needed
try:
    dxl_io = pypot.dynamixel.DxlIO(port)
except Exception as e:
    print(f"Error opening port {port}: {e}")
    
found_ids = dxl_io.scan()
if len(found_ids) < 2:
    raise IOError('You should connect at least two motors on the bus for this test.')
ids = found_ids[:2]
dxl_io.enable_torque(ids)
speed = dict(zip(ids, itertools.repeat(200)))
dxl_io.set_moving_speed(speed)
print(speed)