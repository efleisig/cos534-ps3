import io
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

import unidecode  # Needs to be installed first (pip install unidecode)
from google.cloud import vision

def _get_labels(image_fpath):
    with io.open(image_fpath, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # image_uri = 'gs://cloud-samples-data/vision/using_curl/shanghai.jpeg'
    # image = vision.Image()
    # image.source.image_uri = image_uri
    client = vision.ImageAnnotatorClient()
    response = client.label_detection(image=image)

    print('Labels (and confidence score):')
    print('=' * 30)
    for label in response.label_annotations:
        print(label.description, '(%.2f%%)' % (label.score * 100.))

    return response.label_annotations

# Part 4A
def get_congress_labels():
    image_folder = "mc_images/"
    all_labels = []
    with open('mc_data_replicated.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["wiki_img_url", "wiki_img_labels", "wiki_img_labelsconf"])
        for f in os.listdir(image_folder):
            labels = _get_labels(image_folder + f)
            # print(labels)

            descriptions = []
            scores = []
            for label in labels:
                descriptions.append(label.description)
                scores.append(str(label.score))
            all_labels.append(descriptions)

            tsv_writer.writerow([f, ", ".join(descriptions), ", ".join(scores)])
    
    # get set of unique labels for problem 
    all_labels_flat = set([elem for pic in all_labels for elem in pic])
    print(all_labels_flat)

# helper function for mapping individuals to genders (used in 4B and 4C)
def _get_genders():
    gender_dict = {}
    with open('mc_data.tsv', 'rt') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        next(tsv_reader)
        for row in tsv_reader:
            # Avoid accent/tilde issues
            gender_dict[unidecode.unidecode(row[9][10:])] = row[2]
    
    return gender_dict

# Part 4B
def get_top_labels():               
    gender_dict = _get_genders()
    print(gender_dict)

    f_counts = {}
    m_counts = {}
    with open('mc_data_replicated.tsv', 'rt') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        next(tsv_reader) # skip the first row, which has headings
        for row in tsv_reader:
            image = unidecode.unidecode(row[0])
            labels = row[1].split(",")
            gender = gender_dict[image]
            for label in labels:
                label = label.strip()
                if gender == "Male":
                    if label in m_counts:
                        m_counts[label] += 1
                    else:
                        m_counts[label] = 1
                else:
                    if label in f_counts:
                        f_counts[label] += 1
                    else:
                        f_counts[label] = 1

    ordered_f = [f for f in f_counts.items() if f[1] >= 5]  # Only consider labels used at least 5 times
    ordered_m = [m for m in m_counts.items() if m[1] >= 5]  # (as done in the paper)

    total_f = sum(value == "Female" for value in gender_dict.values())
    total_m = len(gender_dict) - total_f
    top_f = [(label, i/total_f*100) for label, i in ordered_f]
    top_m = [(label, i/total_m*100) for label, i in ordered_m]

    # Get occurrences of gender A's top labels in gender B
    for index, label in enumerate(top_f):
        m_prob = 0
        if label[0] in m_counts:
            m_prob = m_counts[label[0]]/total_m*100

        chi2, p = chisquare([label[1], m_prob])
        top_f[index] = [label[0], label[1], m_prob, chi2]

    for index, label in enumerate(top_m):
        f_prob = 0
        if label[0] in f_counts:
            f_prob = f_counts[label[0]]/total_f*100

        chi2, p = chisquare([label[1], f_prob])
        top_m[index] = [label[0], label[1], f_prob, chi2]

    # Get the top 25 labels by chi2 where occurrence is higher than expected for that gender
    top_f = [f for f in sorted(top_f, key=lambda item: item[3]) if f[1] > f[2]][-25:]
    top_m = [m for m in sorted(top_m, key=lambda item: item[3]) if m[1] > m[2]][-25:]

    top_f = sorted(top_f, key=lambda item: item[1])
    top_m = sorted(top_m, key=lambda item: item[1])

    fig, ax = plt.subplots()
    x = np.arange(len(top_f))
    width = 0.35
    rects1 = ax.barh(x - width / 2, [r[1] for r in top_f], width, label='Women')
    rects2 = ax.barh(x + width / 2, [r[2] for r in top_f], width, label='Men')
    ax.set_xlabel('% receiving each label')
    ax.set_title('Top labels for images of women')
    ax.set_yticks(x)
    ax.set_yticklabels([r[0] for r in top_f])
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    x = np.arange(len(top_m))
    width = 0.35
    rects1 = ax.barh(x - width / 2, [r[2] for r in top_m], width, label='Women')
    rects2 = ax.barh(x + width / 2, [r[1] for r in top_m], width, label='Men')
    ax.set_xlabel('% receiving each label')
    ax.set_title('Top labels for images of men')
    ax.set_yticks(x)
    ax.set_yticklabels([r[0] for r in top_m])
    ax.legend()
    plt.show()

# Part 4C
def get_category_means():
    # map all labels to categories
    labels_cats = {}
    with open('labels_categories.csv', 'rt', encoding='utf-8-sig') as in_file:
        for row in csv.reader(in_file):
            labels_cats[row[0].strip()] = row[1]

    # determine who is which gender
    gender_dict = _get_genders()

    # get total counts for each category, then can just divide by the number of people
    m_counts = {} 
    f_counts = {} 
    categories = ['physicaltrait_body', 'clothing_apparel', 'color_adjective', 'occupation', 'other']
    for category in categories:
        m_counts[category] = 0
        f_counts[category] = 0

    with open('mc_data_replicated.tsv', 'rt') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        next(tsv_reader) # skip the first row, which has headings
        for row in tsv_reader:
            image = unidecode.unidecode(row[0])
            labels = row[1].split(",")
            gender = gender_dict[image]
            for label in labels:
                label = label.strip().lower()
                category = labels_cats[label]
                if gender == "Male":
                    m_counts[category] += 1
                else:
                    f_counts[category] += 1
        
        print(m_counts)
        print(f_counts)

        # get the mean counts
        total_m = sum(x == 'Male' for x in gender_dict.values())
        total_f = sum(x == 'Female' for x in gender_dict.values())

        mean_m_counts = {k: v / total_m for k, v in m_counts.items()}
        mean_f_counts = {k: v / total_f for k, v in f_counts.items()}

        print("Male: " + str(mean_m_counts))
        print("Female: " + str(mean_f_counts))


if __name__ == '__main__':
    get_congress_labels() # Creates mc_data_replicated.tsv (part 4A)
    get_top_labels() # Creates graphs showing gendered labels (part 4B)
    get_category_means() # Calculates mean label counts (part 4C)

