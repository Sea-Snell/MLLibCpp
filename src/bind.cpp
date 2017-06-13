#include "Node.hpp"
#include "pybind11/stl_bind.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(vector<double>);
PYBIND11_MAKE_OPAQUE(vector<Node*>);
PYBIND11_MAKE_OPAQUE(vector<NumObject>);

PYBIND11_PLUGIN(MLLibCpp) {
    py::module m("MLLibCpp", "Machine Learning Library");

    py::bind_vector<vector<double>>(m, "Vector");
    py::bind_vector<vector<Node*>>(m, "NodeVector");
    py::bind_vector<vector<NumObject>>(m, "NumObjectVector");

    py::class_<NumObject>(m, "NumObject")
    	.def_readonly("rank", &NumObject::rank)
    	.def_readonly("dimentions", &NumObject::dimentions)
    	.def_readonly("values", &NumObject::values)
    	.def(py::init<>())
    	.def(py::init<double &>())
    	.def(py::init<vector<double> &, vector<double> &>())
    	.def(py::init<vector<double> &, double &>())
    	.def(py::init<vector<double> &>())
    	.def("describe", &NumObject::describe)
    	.def("getIndex", &NumObject::getIndex)
    	.def("setIndex", &NumObject::setIndex)
    	.def("__str__", &NumObject::describe)
    	.def("__repr__", &NumObject::describe);

    py::class_<Node>(m, "Node")
        .def_readwrite("inputs", &Node::inputs)
        .def_readwrite("name", &Node::name)
        .def_readonly("derivativeMemo", &Node::derivativeMemo, py::return_value_policy::reference)
        .def("getValue", &Node::getValue, py::return_value_policy::reference)
        .def("describe", &Node::describe)
        .def("derive", &Node::derive)
        .def("memoize", &Node::memoize, py::return_value_policy::reference)
        .def("__str__", &Node::describe)
        .def("__repr__", &Node::describe);


    py::class_<Constant, Node>(m, "Constant")
        .def_readwrite("value", &Constant::value, py::return_value_policy::reference)
        .def(py::init<NumObject &>());

    py::class_<Variable, Constant>(m, "Variable")
        .def_readwrite("derivative", &Variable::derivative, py::return_value_policy::reference)
        .def(py::init<NumObject &>());

    py::class_<BasicOperator, Node>(m, "BasicOperator")
        .def("operation", &BasicOperator::operation);

    py::class_<Add, BasicOperator>(m, "Add")
        .def(py::init<Node*, Node*>());

        

    return m.ptr();
}