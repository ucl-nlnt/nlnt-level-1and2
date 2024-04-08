# nlnt-level-1and2
Training and script processing for NLNT level 1 and 2.

Current iteration uses 3000 samples using an automated script.
Models will be posted in a Google Drive folder once trained.

# Data packet structure
    {
    username : ...,
    natural_language_prompt : ...,
    timestamp_s : ...,
    timestamp_float : ...,
    ground_truth : ...,
    simulation : int [optional],
    states : [
        {
            laser_scan : {
                None
                NOTE: currently unsupported in level 1 and 2 simulated data 
            },

            twist : {
                linear : [x, y, z] # usually x is the only non-zero
                angular : [x, y, z] # usually z is the only non-zero
            },

            imu : {
                quarternion_orientation : [...],
                orientation_covariance : [...],
                angular_velocity : [x, y ,z],
                angular_velocity_covariance : [...],
                linear_acceleration : [...],
                linear_acceleration_covariance : [...]
            },

            odometry : {
                time_sec : float,
                time_nano : float,
                pose_position : [x, y, z],
                pose_orientation_quarternion : [x, y, z, w],
                object_covariance : float array [...] # usually not useful
            },

            battery : { # not usually useful
                ...
            },

            frame_data : np.ndarray # turtlebot 3 camera image; NOTE: None for simulated as of april 8, 2024
            distance_traveled : float,
            radians_rotated : float,

        },
        {
        ...
        },
        ...
    ]


    }
"""
