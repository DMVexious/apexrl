# models/

The car, tracks and scenery are generated procedurally at runtime (`src/entities/Car.js`,
`src/entities/Tracks.js`) from `shared/sim_config.json`, so no GLTF assets are required.

Track geometry *must* stay procedural: the physics colliders, the sensor rays and the training
simulator all derive from the same centerline spec. Drop decorative `.glb` files here if you want
to swap in custom visuals for the car body.
