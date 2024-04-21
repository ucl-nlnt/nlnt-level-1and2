# nlnt-level-1and2
Training and script processing for NLNT level 1 and 2.

Current iteration uses 21K dataset samples using an automated script.
Models will be posted in a Google Drive folder once trained.

Current iteration will uses Mistral 7B.
Will use new Llama 3 once the ff. is completed and processed:
   
    1.) Bogus commands        (in progress)
    
    2.) Prompt rephrasing     (in progress)
    
To add the ff to the dataset:
    
    1.) Lidar-based obstacle detection and avoidance.
    
    2.) Early "giving up", i.e. task is deemed to be impossible during execution
        
        - Example: moving one meter past a solid brick wall.
    
    3.) Real-life dataset;

# Data packet structure
    {
    username : ...,
    natural_language_prompt : ...,
    generated_rephrasal : ...,             # NEW! Created after plugging into Llama 3; MAY NOT ALWAYS EXIST.
    timestamp_s : ...,
    timestamp_float : ...,
    ground_truth : ...,
    simulation : int [optional],
    states : [                             # List of frames
        {
            laser_scan : {
                None                       # NOTE: currently unsupported in level 1 and 2 simulated data 
            },

            twist : {
                linear : [x, y, z]         # usually x is the only non-zero
                angular : [x, y, z]        # usually z is the only non-zero
                time : float               # may not exist in older versions
            },

            imu : {                        # may be useful to extrapolate how much total distance the robot has moved
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
