"""
Module for reading inputs from the DFTTK MongoDB database.
"""

# Standard library imports
import os
import numpy as np

# Third-party imports
import pandas as pd
from pymongo import MongoClient

# DFTTK imports
import dfttk.eos_fit as eos_fit

class MongoDBReader:
    def __init__(self, connection_string:str , db: str="DFTTK", collection: str="community"):
        self.cluster = MongoClient(connection_string)
        self.db = self.cluster[db]
        self.collection = self.db[collection]
        
    def get_object_ids(self, reduced_formula: str):
        query = {"material.reduced formula": reduced_formula}
        matching_docs = self.collection.find(query)
        object_ids = [doc["_id"] for doc in matching_docs]
        
        return object_ids
    
    def get_documents(self, object_ids: list):
        documents = []
        for object_id in object_ids:
            documents.append(self.collection.find_one ({"_id": object_id}))
        
        return documents
    
    # TODO: Add the possibility of including all EOS fits
    def get_f_plus_pv(self, documents: list[dict]):
        
        # Assuming all the temperatures and volumes are the same
        temperature = documents[0]["properties"]["QHA phonons"]["temperature"]
        min_temperature = temperature["min"]
        max_temperature = temperature["max"]
        dT = temperature["dT"]
        temperature_range = np.arange(min_temperature, max_temperature + dT, dT)

        volume = documents[0]["properties"]["QHA phonons"]["volume range"]
        min_volume = volume["min"]
        max_volume = volume["max"]
        volume_range = np.linspace(min_volume, max_volume, 1000)
            
        f_plus_pv = np.zeros((len(temperature_range), len(volume_range), len(documents)))
        
        for index, document in enumerate(documents):   
            eos_name = document["properties"]["QHA phonons"]["EOS fit"]["EOS"]
            eos_name = eos_name + "_equation"
            eos_constants_list = document["properties"]["QHA phonons"]["EOS fit"]["constants"]

            eos_function = getattr(eos_fit, eos_name)
            for row, eos_constants in enumerate(eos_constants_list):
                a = eos_constants["a"]
                b = eos_constants["b"]
                c = eos_constants["c"]
                d = eos_constants["d"]
                e = eos_constants["e"]
                
                f_plus_pv[row, :, index] = eos_function(volume_range, a, b, c, d)
        
        return f_plus_pv, temperature_range, volume_range
    
    def get_cv_vib(self, documents: list[dict]):
        temperature = documents[0]["properties"]["Fvib phonons"]["temperature"]
        min_temperature = temperature["min"]
        max_temperature = temperature["max"]
        dT = temperature["dT"]
        temperature_range = np.arange(min_temperature, max_temperature + dT, dT)
        
        volume = documents[0]["properties"]["Fvib phonons"]["volume range"]
        min_volume = volume["min"]
        max_volume = volume["max"]
        volume_range = np.linspace(min_volume, max_volume, 1000)
    
    
    

        