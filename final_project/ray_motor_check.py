import ray
import time
import itertools

@ray.remote
class HardwareActor:
    def __init__(self, port='COM7'):
        import pypot.dynamixel
        try:
            self.dxl_io = pypot.dynamixel.DxlIO(port)
        except Exception as e:
            raise Exception(f"Error opening port {port}: {e}")

        found_ids = self.dxl_io.scan()
        if len(found_ids) < 2:
            raise IOError('You should connect at least two motors on the bus for this test.')
        self.ids = found_ids[:2]
        self.dxl_io.enable_torque(self.ids)
        speed = dict(zip(self.ids, itertools.repeat(200)))
        self.dxl_io.set_moving_speed(speed)

    def set_goal_position(self, pos):
        self.dxl_io.set_goal_position(pos)

    def get_present_position(self):
        return self.dxl_io.get_present_position(self.ids)

    def reset_positions(self):
        self.dxl_io.set_goal_position({self.ids[0]: 0, self.ids[1]: 0})

    def close(self):
        self.dxl_io.close()

def main():
    ray.init()

    # Instantiate the HardwareActor
    hardware_actor = HardwareActor.remote(port='COM7')

    try:
        # Reset the motor positions
        ray.get(hardware_actor.reset_positions.remote())
        print("Motors reset to initial positions.")

        # Define goal positions for the motors
        goal_positions = {0: 0, 1: 0}
        ray.get(hardware_actor.set_goal_position.remote(goal_positions))
        print(f"Set goal positions to: {goal_positions}")

        # Wait for a while to let motors reach the goal position
        time.sleep(2)

        # Get the current motor positions
        current_positions = ray.get(hardware_actor.get_present_position.remote())
        print(f"Current motor positions: {current_positions}")

    finally:
        # Close the hardware actor
        ray.get(hardware_actor.close.remote())
        ray.shutdown()

if __name__ == "__main__":
    main()
