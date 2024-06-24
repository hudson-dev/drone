from KneronDrone import KneronDrone

drone_run = True
use_controller = False
num = 0;

kneronDrone = KneronDrone(drone_run, use_controller)

kneronDrone.init_kneron();
kneronDrone.init_drone();

print("-----------------------running drone!-----------------------")

# Running the drone
while (drone_run):
    kneronDrone.keyboard_input()
    kneronDrone.run_inference()

    # # controller controls
    # if (use_controller):
    #     kneronDrone.controller_input()
