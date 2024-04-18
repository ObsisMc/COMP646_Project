import torch
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine
from PIL import Image
import ast  # For converting string representations of lists into lists

class CLIPModelWrapper:
    class Clip:
        def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda", csv_file='../image_embeddings.csv'):
            """
            Initializes a Clip object.

            Args:
                model_name (str, optional): The name or path of the pre-trained CLIP model. Defaults to "openai/clip-vit-base-patch32".
                device (str, optional): The device to use for running the model. Defaults to "cuda".
                csv_file (str, optional): The path to the CSV file containing image embeddings. Defaults to '../image_embeddings.csv'.
            """
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = device
            df = pd.read_csv(csv_file)
            df['embedding'] = df['embedding'].apply(ast.literal_eval)
            self.embeddings_df = df

    def get_text_embedding(self, prompt):
            """
            Get the text embedding for a given prompt.

            Args:
                prompt (str): The text prompt.

            Returns:
                numpy.ndarray: The text embedding as a flattened numpy array.
            """
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            return text_features.cpu().numpy().flatten()
    

    def find_top_similar_images(self, prompt, top_n=5):
        """
        Finds the top N similar images based on a given prompt.

        Parameters:
            prompt (str): The prompt to compare the images against.
            top_n (int): The number of top similar images to return. Default is 5.

        Returns:
            list: A list of file names of the top N similar images.
        """
        prompt_embedding = self.get_text_embedding(prompt)

        similarities = []
        for img_emb in self.embeddings_df['embedding']:
            img_emb_array = np.array(img_emb).flatten()
            similarity = 1 - cosine(prompt_embedding, img_emb_array)
            similarities.append(similarity)

        top_indices = np.argsort(similarities)[::-1][:top_n]  # Get indices of top N similarities

        return self.embeddings_df.iloc[top_indices]['image_file_name'].tolist()
