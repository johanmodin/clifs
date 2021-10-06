import os
import clip
import torch
import cv2
import math
import glob
import logging
import time

from PIL import Image
import numpy as np
from patchify import patchify

class CLIFS:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)


        # Choose device and load the chosen model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(os.getenv('MODEL'), self.device, jit=False)

        self.image_features = None
        self.feature_idx_to_video = []

        # Preload the videos in the data input directory
        # This is done as upload through web interface isn't implemented yet
        for f in glob.glob('{}/*'.format(os.getenv('INPUT_DIR'))):
            self.add_video(f)


    def add_video(self, path, batch_size=128, ms_between_features=1000,
                  patch_size=360):
        # Calculates features from video images.
        # Loops over the input video to extract every frames_between_features
        # frame and calculate the features from it. The features are saved
        # along with mapping of what video and frame each detection
        # corresponds to.
        # The actual batch size can be up to batch_size + number of patches.
        logging.info('Adding video: {}'.format(path))
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_between_features = fps / (1000 / ms_between_features)
        feature_list = []
        feature_video_map = []

        frame_idx = 0
        to_encode = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frames_between_features == 0:
                patches = self._make_patches(frame, patch_size) + [frame]
                for idx, patch in enumerate(patches):
                    feature_data = {'video_path': path,
                                    'frame_idx': frame_idx,
                                    'time': frame_idx / fps}
                    feature_video_map.append(feature_data)
                    to_encode.append(patch)
            if len(to_encode) >= batch_size:
                image_features = self._calculate_images_features(to_encode)
                feature_list.append(image_features)
                to_encode = []

            frame_idx += 1
        if len(to_encode) > 0:
            image_features = self._calculate_images_features(to_encode)
            feature_list.append(image_features)
        feature_t = torch.cat(feature_list, dim=0)
        self._add_image_features(feature_t, feature_video_map)


    def _make_patches(self, frame, patch_size):
        # To get more information out of images, we divide the image
        # into smaller patches that are closer to the input size of the model
        step = int(patch_size / 2)
        patches_np = patchify(frame, (patch_size, patch_size, 3),
                              step=step)
        patches = []
        for i in range(patches_np.shape[0]):
            for j in range(patches_np.shape[1]):
                patches.append(patches_np[i, j, 0])
        return patches


    def _calculate_images_features(self, images):
        # Preprocess an image, send it to the computation device and perform
        # inference
        logging.info(f'Calculating features for batch of {len(images)} frames')
        for i in range(len(images)):
            t1 = time.time()
            images[i] = self._preprocess_image(images[i])
        t1 = time.time()
        image_stack = torch.stack(images, dim=0)
        image_t = image_stack.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_t)
        return image_features


    def _preprocess_image(self, image):
        # cv2 image to PIL image to the model's preprocess function
        # which makes sure the image is ok to ingest and makes it a tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return self.preprocess(image)


    def _add_image_features(self, new_features, feature_video_map):
        # Controls the addition of image features to the object
        # such that video mappings are provided, etc.,
        assert(new_features.shape[0] == len(feature_video_map))
        new_features /= new_features.norm(dim=-1, keepdim=True)
        if self.image_features is not None:
            self.image_features = torch.cat((self.image_features, new_features),
                                            dim=0)
        else:
            self.image_features = new_features
        self.feature_idx_to_video.extend(feature_video_map)


    def search(self, query, n=9, threshold=37):
        # Takes a query, calculates its features and finds the most similar
        # image features and thus corresponding images
        text_inputs = torch.cat([clip.tokenize(query)]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_features @ self.image_features.T)

        # The 100 in n * 100 is just an arbitrary value to 1) find enough
        # matches to fill the results while 2) still being able to filter
        # the matches that stem from the same frame through the image patches
        # This could be made more efficient through keeping track of the indices
        # and matching these with the metadata in self.feature_idx_to_video
        values, indices = similarity[0].topk(n * 100)

        used_images = set()
        response_matches = []
        for indices_idx, similarity_idx in enumerate(indices):
            if len(response_matches) >= n:
                break
            initial_match_data = self.feature_idx_to_video[similarity_idx]
            score = float(values[indices_idx].cpu().numpy())
            img_hash = '{}-{}'.format(initial_match_data['video_path'],
                                      initial_match_data['frame_idx'])
            if img_hash in used_images:
                continue

            if score < threshold:
                # We've reached the point in the sorted list
                # where scores are too low
                if len(response_matches) == 0:
                    logging.info('No matches with score >= threshold found')
                break

            image_path = self._write_image_from_match(initial_match_data)
            if image_path is None:
                continue
            full_match_data = {
                               **initial_match_data,
                               'score': score,
                               'image_path': image_path,
                              }
            logging.info('Frame ({}): {}'.format(query, full_match_data))
            response_matches.append(full_match_data)
            used_images.add(img_hash)
        return response_matches


    def _write_image_from_match(self, match):
        path, ext = os.path.splitext(match['video_path'])
        video_name = os.path.splitext(os.path.basename(path))[0]
        image_name = '{}-{}.jpg'.format(video_name, match['frame_idx'])
        image_path = '{}/{}'.format(os.getenv('OUTPUT_DIR'), image_name)
        if os.path.exists(image_path):
            return image_name
        cap = cv2.VideoCapture(match['video_path'])
        cap.set(cv2.CAP_PROP_POS_MSEC, match['time'] * 1000)
        ret, frame = cap.read()
        if not ret:
            return None
        cv2.imwrite(image_path, frame)
        return image_name
