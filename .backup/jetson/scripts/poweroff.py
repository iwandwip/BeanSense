#!/usr/bin/env python3
import os

def power_off():
    os.system("/sbin/poweroff")

if __name__ == "__main__":
    power_off()