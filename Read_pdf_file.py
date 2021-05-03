!pip install tika

## Import
from tika import parser
import sys
import os

## Get the input from user
user_input = input("Enter the path of your file: ")
text = parser.from_file(user_input)['content']
