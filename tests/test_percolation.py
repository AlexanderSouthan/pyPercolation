#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyPercolation.percolation import percolation


class TestRandomWalk(unittest.TestCase):

    def test_random_walk(self):
        percol_test = percolation()
