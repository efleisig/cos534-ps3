import io
import os
import csv

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
    print(ordered_f)
    total_m = len(gender_dict) - total_f
    ordered_f = [(label, i/total_f*100) for label, i in ordered_f]
    ordered_m = [(label, i/total_m*100) for label, i in ordered_m]
    print(ordered_f)




if __name__ == '__main__':
    #get_congress_labels()
    get_top_labels()

