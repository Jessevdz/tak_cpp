#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/script.h>
#include "env.cpp"
#include "board.h"
#include "board.cpp"
#include "util.cpp"

namespace py = pybind11;
// constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(tak_cpp, m)
{
    py::class_<TakEnv>(m, "TakEnv")
        .def(py::init<torch::jit::script::Module>())
        .def("play_game", &TakEnv::play_game)
        .def("write_experience_to_cout", &TakEnv::write_experience_to_cout);
}