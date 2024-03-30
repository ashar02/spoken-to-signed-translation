import os
from collections import defaultdict
from typing import List
from pymongo import MongoClient, InsertOne
from dotenv import load_dotenv
import time
import threading
from datetime import datetime

from pose_format import Pose

from spoken_to_signed.text_to_gloss.types import Gloss


class PoseLookup:
    def __init__(self, rows: List, directory: str = None):
        self.directory = directory

        load_dotenv()
        connection_string = os.getenv('DB_URI')
        if connection_string:
            self.client = MongoClient(connection_string)
            while True:
                try:
                    self.client = MongoClient(connection_string)
                    self.db = self.client['translate']
                    print("Connected to the database.")
                    break
                except Exception as e:
                    print(f"Failed to connect to the database. Retrying in 5 seconds... Error: {e}")
                    time.sleep(5)

        self.words_index = self.make_dictionary_index(rows, based_on="words")
        self.glosses_index = self.make_dictionary_index(rows, based_on="glosses")

        self.file_systems = {}

    def make_dictionary_index(self, rows: List, based_on: str):
        # As an attempt to make the index more compact in memory, we store a dictionary with only what we need
        languages_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for d in rows:
            lower_term = d[based_on].lower()
            languages_dict[d['spoken_language']][d['signed_language']][lower_term].append({
                "path": d['path'],
                "start": d['start'],
                "end": d['end'],
            })
        return languages_dict

    def read_pose(self, pose_path: str):
        if pose_path.startswith('gs://'):
            if 'gcs' not in self.file_systems:
                import gcsfs
                self.file_systems['gcs'] = gcsfs.GCSFileSystem(anon=True)

            with self.file_systems['gcs'].open(pose_path, "rb") as f:
                return Pose.read(f.read())

        if pose_path.startswith('https://'):
            raise NotImplementedError("Can't access pose files from https endpoint")

        if self.directory is None:
            raise ValueError("Can't access pose files without specifying a directory")

        pose_path = os.path.join(self.directory, pose_path)
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        lookup_list = [
            (self.words_index, (spoken_language, signed_language, word)),
            (self.glosses_index, (spoken_language, signed_language, word)),
            (self.glosses_index, (spoken_language, signed_language, gloss)),
        ]

        for dict_index, (spoken_language, signed_language, term) in lookup_list:
            if spoken_language in dict_index:
                if signed_language in dict_index[spoken_language]:
                    lower_term = term.lower()
                    if lower_term in dict_index[spoken_language][signed_language]:
                        rows = dict_index[spoken_language][signed_language][lower_term]
                        # TODO maybe perform additional string match, for correct casing
                        return self.read_pose(rows[0]["path"])

        raise FileNotFoundError

    def lookup_db(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        collection = self.db['lexicals']
        query = {
            'spoken': spoken_language,
            'signed': signed_language,
            'text': gloss
        }
        document = collection.find_one(query)
        if document:
            pose_data = document['pose']
            return Pose.read(pose_data)
        raise FileNotFoundError

    def lookup_sequence(self, glosses: Gloss, spoken_language: str, signed_language: str, source: str = None):
        poses: List[Pose] = []
        for word, gloss in glosses:
            try:
                pose = self.lookup(word, gloss, spoken_language, signed_language)
                poses.append(pose)
            except FileNotFoundError:
                pass

        if len(poses) == 0:
            gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")

        return poses

    def lookup_sequence_db(self, glosses: Gloss, spoken_language: str, signed_language: str, source: str = None):
        lexicals = self.db['lexicals']
        unique_glosses = {gloss for _, gloss in glosses}
        query = {
            "$or": [
                {
                    "spoken": spoken_language,
                    "signed": signed_language,
                    "text": gloss
                } for gloss in unique_glosses
            ]
        }
        #print('mongo start')
        documents = list(lexicals.find(query))
        #print('mongo end')
        poses: List[Pose] = []
        poses_not_found: List[str] = []
        for word, gloss in glosses:
            found = False
            for document in documents:
                if document['text'] == gloss:
                    poses.append(Pose.read(document['pose']))
                    found = True
                    break
            if not found:
                poses_not_found.append(gloss)
        if poses_not_found:
            self.insert_not_found_glosses_async(poses_not_found, spoken_language, signed_language)
        if len(poses) == 0:
            gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")

        return poses

    def insert_not_found_glosses_async(self, glosses: List[str], spoken_language: str, signed_language: str):
        def run():
            not_found_lexicals = self.db['not_found_lexicals']
            bulk_operations = [
                InsertOne({
                    "spoken": spoken_language,
                    "signed": signed_language,
                    "text": gloss,
                    "createdAt": datetime.now()
                }) for gloss in glosses
            ]
            try:
                not_found_lexicals.bulk_write(bulk_operations, ordered=False)
            except Exception as e:
                print(f"Error inserting glosses: {e}")

        #run()
        threading.Thread(target=run).start()