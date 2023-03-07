#include <pybind11/pybind11.h>
#include "env.cpp"
#include "board.h"
#include "board.cpp"
#include "util.cpp"
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(tak_cpp, m)
{
    py::class_<TakEnv>(m, "TakEnv")
        .def(py::init<>())
        .def("play_game", &TakEnv::play_game, byref)
        .def("load_player", &TakEnv::load_player);
}