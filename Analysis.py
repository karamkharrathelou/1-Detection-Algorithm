import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import h5py
from scipy import ndimage
import os
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
import seaborn as sns
from datetime import datetime, timedelta

def make_ci(t,y, ci):
    nptsn=int(np.floor(len(y)/ci))
    yn=np.empty(nptsn)
    for i in range(0,nptsn):
        yn[i]=np.mean(y[i*ci:i*ci+ci-1])
    return yn
def find_fminmax(values, threshold, consecutive_count):
    # Check if each value exceeds the threshold
    binary_values = [1 if val > threshold else 0 for val in values]
    # Find the indices of the first occurrence of consecutive_count consecutive ones
    first_occurrence_index = find_consecutive_ones(binary_values, consecutive_count)
    # Reverse the list and find the indices of the last occurrence of consecutive_count consecutive ones
    reversed_binary_values = binary_values[::-1]
    last_occurrence_index_reverse = find_consecutive_ones(reversed_binary_values, consecutive_count)
    # If there are no consecutive_count consecutive ones, return -1
    if first_occurrence_index == -1 or last_occurrence_index_reverse == -1:
        return -1, -1
    # Calculate the actual indices in the original list
    last_occurrence_index = len(values) - 1 - last_occurrence_index_reverse
    return first_occurrence_index, last_occurrence_index
def find_consecutive_ones(binary_values, consecutive_count):
    for i in range(len(binary_values) - consecutive_count + 1):
        if all(binary_values[i + j] == 1 for j in range(consecutive_count)):
            return i
    return -1
def radar_date(timestamp):
    # Convert the timestamp to a datetime object using the radar epoch
    dt_object = datetime.utcfromtimestamp(timestamp)
    # Extract individual date components
    year = dt_object.year
    month_name = dt_object.strftime('%B')  # Get full month name
    day = dt_object.day
    # Extract individual time components
    hours = dt_object.hour
    minutes = dt_object.minute
    seconds = dt_object.second

    # Return two values
    return (year, month_name, day), (hours, minutes, seconds)

def smooth(y,box_pts):
    box=np.ones(box_pts)/box_pts
    y_smooth=np.convolve(y,box,mode="same")
    return y_smooth
fmin_values, fmax_values, Amin_values, Amax_values,time_values=[], [], [], [],[]
def combine_lists(list1, list2, list3, list4,list5):
    # Combine the lists into a matrix
    matrix = np.column_stack((list1, list2, list3, list4,list5))
    return matrix
def highlight_most_repeated(ax, values, color, histogram_name, x_label, y_label):
    counts, bins, patches = ax.hist(values, bins=20, color=color, alpha=0.7, edgecolor='black')
    most_repeated_value = bins[np.argmax(counts)]
    median_value = np.median(values)
    ax.axvline(x=most_repeated_value, color='red', linestyle='-', linewidth=2, label='Most Repeated Value')
    ax.axvline(x=median_value, color='green', linestyle='--', linewidth=2, label='Median Value')
    ax.legend()
    num_elements = len(values)
    title_text = f'{histogram_name} Histogram\nMost Repeated Value: {most_repeated_value:.2f}, Median: {median_value:.2f},samples number: {num_elements}'
    ax.set_title(title_text, fontsize='large', fontweight='bold')
    ax.set_xlabel(x_label, fontsize='medium')
    ax.set_ylabel(y_label, fontsize='medium')
    ax.tick_params(axis='both', labelsize='medium')
    ax.grid(True, linestyle='--', alpha=0.7)
    sns.despine()

    # Add data labels
    for count, bin_edge, patch in zip(counts, bins[:-1], patches):
        if count > 0:
            x = bin_edge + (bins[1] - bins[0]) / 2
            y = count
            ax.text(x, y, f'{int(count):,}', ha='center', va='bottom', fontsize='small', color='black')

    # Make bars slightly transparent
    for patch in patches:
        patch.set_alpha(0.8)

def make_fft(t,y):
    dt = t[1]-t[0] # dt -> temporal resolution ~ sample rate
    f=np.fft.fftfreq(t.size, dt) # frequency axis
    Y=np.fft.fft(y)   # FFT
    f=np.fft.fftshift(f)
    Y=np.fft.fftshift(Y)/(len(y))
    return f,Y

def find_objects( det_mask):
        obj_labels, _ = ndimage.label(det_mask, np.ones((3, 3)))
        obj_slices = ndimage.find_objects(obj_labels)
        return obj_slices

def make_ifft(f, Y, dt):
    f = np.fft.ifftshift(f)  # Undo the frequency shift
    Y = np.fft.ifftshift(Y)  # Undo the FFT shift
    y = np.fft.ifft(Y)*len(Y)# Perform the inverse FFT
    N = len(y)  # Length of the time-domain signal
    t = np.arange(N) * dt
    return t, y
def combine_inboxes(boxes):
    result = []

    for i, box1 in enumerate(boxes):
        combined = False

        for j, box2 in enumerate(boxes):
            if i != j:  # Avoid comparing a box to itself
                if (
                    box1[0] >= box2[0]
                    and box1[1] <= box2[1]
                    and box1[2] >= box2[2]
                    and box1[3] <= box2[3]
                ):
                    # box1 is inside box2
                    combined = True
                    break

        if not combined:
            result.append(box1)

    return result
# Function to calculate the area of a box
def calculate_area(box):
    return (box[1] - box[0]) * (box[3] - box[2])

def filter_boxes(boxes, width_threshold, height_threshold):
    filtered_boxes = [
        box for box in boxes if (box[1] - box[0]) >= width_threshold and (box[3] - box[2]) >= height_threshold
    ]
    return filtered_boxes
# Function to find the biggest 4 boxes and combine their dimensions
def biggest_boxes(boxes,noDP):
    # Check if there are at least 3 boxes
    if not boxes:
        return [], [int(noDP/2), int(noDP/2), 0, 0]

    # Sort boxes based on area in descending order
    sorted_boxes = sorted(boxes, key=calculate_area, reverse=True)
    if len(sorted_boxes)<3:
        biggest_boxes = sorted_boxes[:]
        
        # Calculate the overall bounding box
        combined_box = [
            min(box[0] for box in biggest_boxes),
            max(box[1] for box in biggest_boxes),
            min(box[2] for box in biggest_boxes),
            max(box[3] for box in biggest_boxes)
        ]
    else:
        biggest_boxes = sorted_boxes[:]
        
        # Calculate the overall bounding box
        combined_box = [
            min(box[0] for box in biggest_boxes),
            max(box[1] for box in biggest_boxes),
            min(box[2] for box in biggest_boxes),
            max(box[3] for box in biggest_boxes)
        ]
    return biggest_boxes, combined_box

def rebuild_statistics(data_directory):
    if not os.path.exists("statistics"):
        os.makedirs("statistics")
    if not os.path.exists("power_profile"):
        os.makedirs("power_profile")
    powerpr = []
    fmin_values, fmax_values, Amin_values, Amax_values,time_values=[], [], [], [],[]
    directory = os.listdir(data_directory)
    # i=0
    # j=0
    for i in range(np.size(directory)):
        # if i==1:
        #     break
        f2 = h5py.File(os.path.join(data_directory, directory[i]), 'r')
        top_items = list(f2.items())
        GI = np.array(f2.get("GI"))
        GI = np.array(GI.tolist())
        noRX = int(GI[0])
        noTX = int(GI[1])
        noRG = int(GI[2])
        noDP = int(GI[3])
        dt = GI[4]
        time = np.array(f2.get("time"))
        time = np.array(time.tolist())
        time_start = int(time[0])
        time_end = int(time[1])
        #j=14
        for j in range(len(top_items) - 2):
            # if j==1:
            #     break
            currenttime = float(top_items[j][0])
            noise_std_dev=0.1
            sub = f2.get(f"{currenttime}")
            SI = np.array(sub.get("SI"))
            SI = np.array(SI.tolist())
            NL = SI[2]
            threshold = SI[3]
            NL=10 ** (NL/ 10)#+10 ** (threshold/ 10)
            data =np.zeros((noRX, noTX, noRG, noDP), dtype=complex)
            data=(np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))*np.sqrt(NL/2)
            sub_items = list(sub.items())
            for k in range(len(sub_items) - 1):
                rge = int(sub_items[k][0])
                rangef = sub.get(f"{rge}")
                RXTX = list(rangef.get("RXTX"))
                index = np.array(rangef.get("index"))
                co = 0
                for l in range(noRX):
                    for m in range(noTX):
                        data[l, m, rge, index] = RXTX[co]
                        co = co + 1
            power = np.mean(np.abs(data) ** 2, axis=(0, 1))
            powerdb = 10 * np.log10(power)
            mvf = np.median(power, 1)# Calculate the mean power in each range gate
            NLl = np.median(mvf)#noise level in linear scale
            NL = 10 * np.log10(NLl)  # Convert NL to dB
            SNRf = 10 * np.log10(power/NLl)# Calculate SNR in dB
            powerprofile = np.sum(powerdb, 1)
            powerpr.append(powerprofile)
            mvf = np.median(powerdb, 1)
            thf = np.median(mvf) + threshold
            f = np.fft.fftfreq(noDP, dt)
            f = np.fft.fftshift(f)
            ranges = np.linspace(0, noRG - 1, noRG)
            ranges=ranges*3#/2
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pp=plt.imshow(SNRf,extent=[min(f),max(f),min(ranges),max(ranges)],origin='lower',aspect="auto",cmap="inferno",vmin=0,vmax=20)
            plt.colorbar(pp,label='SNR/dB.')
            plt.xlabel('Frequency/Hz')
            plt.title('SNR')
            plt.ylabel('Total range/Km')
            plt.savefig(f"{currenttime}.png", dpi=300)
            plt.show()
data_directory =os.path.join(os.getcwd(),'data')
rebuild_statistics(data_directory)
