add_library( StaticNet
    "include/activation.h"
    "include/backward_prop.h"
    "include/forward_prop.h"
    "include/loss.h"
    "include/matrix.h" "src/matrix.cpp"
    "include/neural_net.h"
    "include/optimize.h"
    "include/propagation_data.h"
    "include/util.h" "src/util.cpp"
)
target_include_directories( StaticNet
    PUBLIC
        "include"
)
