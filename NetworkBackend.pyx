#cython: boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, nonecheck=False, cdivision=True

from zmq.backend.cython.socket cimport Socket
from zmq.backend.cython import constants


cpdef forward_response(Socket backend, Socket frontend):
    cdef bytes client
    cdef bytes response

    while True:
        client = backend.recv()
        response = backend.recv()

        frontend.send(client, constants.SNDMORE)
        frontend.send(response)