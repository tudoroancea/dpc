# car mass and geometry
m = 230.0  # mass
wheelbase = 1.5706  # distance between the two axles
car_length = 2.0
car_width = 1.0
# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784
# actuator limits
T_max = 500.0
delta_max = 0.5
# general OCP parameters
Nf = 40  # horizon size
nx = 4  # state dimension
nu = 2  # control dimension
dt = 1 / 20  # sampling time
