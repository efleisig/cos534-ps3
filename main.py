import io
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

from google.cloud import vision

def get_labels(image_fpath):
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

def get_congress_labels():
    image_folder = "mc_images/"
    with open('mc_data_replicated.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["wiki_img_url", "wiki_img_labels", "wiki_img_labelsconf"])
        for f in os.listdir(image_folder):
            labels = get_labels(image_folder + f)

            descriptions = []
            scores = []
            for label in labels:
                descriptions.append(label.description)
                scores.append(str(label.score))

            tsv_writer.writerow([f, ", ".join(descriptions), ", ".join(scores)])

def get_top_labels():
    import unidecode                    # Needs to be installed first (pip install unidecode)

    gender_dict = {}
    with open('mc_data.tsv', 'rt') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        next(tsv_reader)
        for row in tsv_reader:
            gender_dict[unidecode.unidecode(row[9][10:])] = row[2] # Avoid accent/tilde issues

    print(gender_dict)

    f_counts = {}
    m_counts = {}
    with open('mc_data_replicated.tsv', 'rt') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        next(tsv_reader)
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

    ordered_f = sorted(f_counts.items(), key=lambda item: item[1])
    ordered_m = sorted(m_counts.items(), key=lambda item: item[1])
    total_f = sum(value == "Female" for value in gender_dict.values())
    total_m = len(gender_dict) - total_f
    top_f = [(label, i/total_f*100) for label, i in ordered_f][-25:]        # This should actually be ordered by
    top_m = [(label, i/total_m*100) for label, i in ordered_m][-25:]        # chi-squared test (todo)

    # Get occurrences of gender A's top labels in gender B
    print(top_f)
    for index, label in enumerate(top_f):
        m_count = 0
        if label[0] in m_counts:
            m_count = m_counts[label[0]]/total_m*100
        top_f[index] = [label[0], label[1], m_count]

    for index, label in enumerate(top_m):
        f_count = 0
        if label[0] in f_counts:
            f_count = f_counts[label[0]]/total_f*100
        top_m[index] = [label[0], label[1], f_count]

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
    rects1 = ax.barh(x - width / 2, [r[1] for r in top_m], width, label='Women')
    rects2 = ax.barh(x + width / 2, [r[2] for r in top_m], width, label='Men')
    ax.set_xlabel('% receiving each label')
    ax.set_title('Top labels for images of men')
    ax.set_yticks(x)
    ax.set_yticklabels([r[0] for r in top_m])
    ax.legend()
    plt.show()





if __name__ == '__main__':
    #get_congress_labels()
    get_top_labels()

