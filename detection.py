import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import h5py
from scipy import ndimage
import os
import shutil
from math import ceil
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
def make_fft(t,y):
    dt = t[1]-t[0] # dt -> temporal resolution ~ sample rate
    f=np.fft.fftfreq(t.size, dt) # frequency axis
    Y=np.fft.fft(y)   # FFT
    f=np.fft.fftshift(f)
    Y=np.fft.fftshift(Y)#/(len(y))
    return f,Y


def find_objects( det_mask):
        # label objects
        obj_labels, _ = ndimage.label(det_mask, np.ones((3, 3)))
        # find objects
        obj_slices = ndimage.find_objects(obj_labels)
        return obj_slices

firsttime = None
time_end=None

def detection(y,th,time_start,ct,mins,m):
    global firsttime
    global time_end
    sub_directory = "data"

    # Check if the directory already exists
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    noRX, noTX, noRG, noDP = y.shape# Extract dimensions of the input data 'y'
    ranges=np.linspace(0,noRG-1,noRG)
    t = np.linspace(0, ct - 1, noDP)  # Create a time vector 't'
    dt = t[1] - t[0]  # Calculate the time resolution
    # Create general information variable 'gi' to store information about the data
    fft = np.zeros(y.shape, dtype=complex)# Initialize an array 'fft' to store FFT results
    for i in range (noRX):# Perform FFT
            for j in range (noTX):
                for k in range (noRG):
                    fi,fft[i,j,k,:]=make_fft(t,y[i,j,k,:])
    power =np.mean(np.abs(fft)**2,axis=(0,1))# Calculate the power spectrum
    mvf = np.median(power, 1)# Calculate the mean power in each range gate
    # Find the minimum mean power to estimate the noise level (NL) in dB
    NLl = np.median(mvf)#noise level in linear scale
    NL = 10 * np.log10(NLl)  # Convert NL to dB
    powerdb = 10*np.log10(power)# Convert the power spectrum to dB scale
    SNRf = 10 * np.log10(power/NLl)# Calculate SNR in dB
    mf = np.mean(SNRf, 0)#removing artifacts
    SNRf = SNRf - mf
    location = np.array(np.where(SNRf>th))# Find the locations where SNR exceeds the threshold
    mask = np.where(SNRf>th, 1, 0)# Create a binary mask to mark the detected locations
    if m<20:
        ranges=ranges*3#/2
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        pp2=ax2.imshow(SNRf,extent=[min(fi),max(fi),min(ranges),max(ranges)],origin='lower',aspect="auto",cmap="inferno",vmin=0,vmax=20)
        plt.colorbar(pp2,label='SNR/dB.')
        plt.xlabel('Frequency/Hz')
        plt.title('SNR')
        plt.ylabel('Total range/Km')
        plt.savefig(f"{m}.png", dpi=300)
        plt.show()
        dataf=SNRf*mask
        slices=find_objects(dataf)
        nonsf=list()
        
        for i in range(np.size(slices,0)):
            
            nonsf.append([slices[i][1].start,slices[i][1].stop,slices[i][0].start,slices[i][0].stop])
        nonsf=combine_inboxes(nonsf)
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        pp2=ax2.imshow(SNRf,extent=[min(fi),max(fi),min(ranges),max(ranges)],origin='lower',aspect="auto",cmap="inferno",
                          interpolation ='nearest',vmin=0,vmax=20)
        plt.colorbar(pp2,label='SNR/dB.')
        for i in range(np.size(nonsf,0)):
            ax2.add_patch(Rectangle((fi[nonsf[i][0]],ranges[nonsf[i][2]]),
                                    fi[nonsf[i][1]-1]-fi[nonsf[i][0]],(ranges[nonsf[i][3]-1]-ranges[nonsf[i][2]]),
                                    fc ='none', 
                                    ec ='g', 
                                    lw = 1))
        plt.xlabel('Frequency/Hz')
        plt.title('SNR')
        plt.ylabel('Total range/Km')
        plt.savefig(f"{time_start}.png", dpi=300)
        plt.show()
    ranges=ranges*2/3
    time_end2 = time_start + ct
    # Create avariable 'si' (special information) to store detection-related information
    si=[int(time_start),int(time_end2),NL,th]
    col_names = ["time start", "time end", "noise level", "threshold"]
    ds_dt = np.dtype( { 'names':col_names,
                         'formats':[ (float), (float), (float), (float), (float)] } )
    si = np.rec.array(si,dtype=ds_dt)
    rang = np.unique(location[0, :])# Identify unique range gate indices where detections occurred
    rounds = int((mins * 60) / ct)
    gi=[noRX,noTX,noRG,noDP,dt]
    col_namesg = ["Receivers number", "Transmitter number","Number of range gates", "Number of data points", "Time resolution"]
    ds_dtg = np.dtype( { 'names':col_namesg,'formats':[ (int), (int), (int), (float), (float)] } )
    gi = np.rec.array(gi,dtype=ds_dtg)
    current_directory = os.getcwd()
    if m== 0 or m % rounds== 0:
        firsttime = time_start
        time_end=time_start+(ct*(rounds-1))
        hdf5_file_path = os.path.join(sub_directory, f"data_{firsttime}_{time_end}")
        with h5py.File(hdf5_file_path, "w") as f2:
            time = np.array([time_start, time_end])
            col_names = ["time start", "time end"]
            ds_dt = np.dtype({'names': col_names, 'formats': [float, float]})
            time_dataset = f2.create_dataset("time", data=time)
            f2.create_dataset("GI", data=gi)
           # f2.create_dataset("frequency",data=fi)
   
    hdf5_file_path = os.path.join(sub_directory, f"data_{firsttime}_{time_end}")
    with h5py.File(hdf5_file_path, "a") as f2:
        f1 = f2.create_group(f"{time_start}")
        f1.create_dataset("SI", data=si)
        
        for r in rang:
            group = f1.create_group(f"{r}")
            index = np.where(mask[r, :] > 0)[0]
            group.create_dataset("index", data=index)
            values = []
            for j in range(noRX):
                for k in range(noTX):
                    value = fft[j, k, r, index]
                    values.append(value)
            group.create_dataset("RXTX", data=values)


def rti(data_directory):
    rti = []
    directory = os.listdir(data_directory)  # Reading the content directory
    for i in range(np.size(directory)):
        f2 = h5py.File(os.path.join(data_directory, directory[i]), 'r')
        top_items = list(f2.items())
        GI = np.array(f2.get("GI"))
        GI = np.array(GI.tolist())
        noRX = int(GI[0])  # Number of receivers.
        noTX = int(GI[1])  # Number of transmitters.
        noRG = int(GI[2])  # Number of range gates.
        noDP = int(GI[3])  # Number of range gates.
        dt = GI[4]  # Time resolution (to restore the frequency axis)
        time = np.array(f2.get("time"))
        time = np.array(time.tolist())
        time_start = int(time[0])
        time_end = int(time[1])  # End time

        for j in range(len(top_items) - 2):
            currenttime = float(top_items[j][0])

            data = np.random.randn(noRX, noTX, noRG, noDP) + 1j * np.random.randn(noRX, noTX, noRG, noDP)

            sub = f2.get(f"{currenttime}")  # Get the general information of the data
            SI = np.array(sub.get("SI"))  # Get the general information of the data
            SI = np.array(SI.tolist())
            NL = SI[2]  # Noise level in dB
            threshold = SI[3]

            sub_items = list(sub.items())
            for k in range(len(sub_items) - 1):
                rge = int(sub_items[k][0])  # For loop needed
                rangef = sub.get(f"{rge}")
                RXTX = list(rangef.get("RXTX"))
                index = np.array(rangef.get("index"))
                co=0
                for l in range(noRX):
                    for m in range(noTX):
                        data[l,m, rge, index] = RXTX[co]
                        co=co+1
            power = np.mean(np.abs(data) ** 2, axis=(0, 1))  # Calculate the power spectrum
            powerdb = 10 * np.log10(power)  # Convert the power spectrum to dB scale
            mvf = np.median(powerdb, 1)  # Mean value for each range
            thf = np.median(mvf) + threshold
            f = np.fft.fftfreq(noDP, dt)
            f = np.fft.fftshift(f)
            ranges = np.linspace(0, noRG - 1, noRG)
            location = np.array(np.where(powerdb > thf))
            mask = np.where(powerdb > thf, 1, 0)
            dataf = powerdb * mask
            slices = find_objects(dataf)
            zz = np.sum(powerdb, 1)
            rti.append(zz)
            print(j)
            

    rti2 = np.array(rti)
    rti2 = rti2.T
    mins = int(np.size(rti2, 1) / 2)
    time = np.linspace(0, 24, np.size(rti2, 1))
    rti2 = rti2 / 10000
    mf = np.mean(rti2, 0)
    rti2 = rti2 - mf

    # Plotting rti and saving data
    if not os.path.exists("rti"):
        os.makedirs("rti")
    rti_path = os.path.join(os.getcwd(),'rti','rti.jpg')
    data_path=os.path.join(os.getcwd(),'rti','rti.npy')
    plt.figure(figsize=(25, 9), dpi=600)
    pp = plt.imshow(rti2, extent=[min(time), max(time), min(ranges), max(ranges)],
                    origin='lower', aspect="auto", cmap="inferno", vmin=0, vmax=2)
    plt.colorbar(pp)
    plt.title('RTI')
    plt.xlabel('Time (hours)')
    plt.ylabel('Ranges')
    plt.savefig(rti_path, dpi=1000,bbox_inches='tight')
    plt.show()
    np.save(data_path,rti2)

def pdd(data_directory):
    directory = os.listdir(data_directory)  # List the files in the given directory
    for i in range(np.size(directory)):
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
        for j in range(len(top_items) - 2):
            currenttime = float(top_items[j][0])
            data = np.random.randn(noRX, noTX, noRG, noDP) + 1j * np.random.randn(noRX, noTX, noRG, noDP)
            sub = f2.get(f"{currenttime}")
            SI = np.array(sub.get("SI"))
            SI = np.array(SI.tolist())
            NL = SI[2]
            threshold = SI[3]
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
            mvf = np.median(powerdb, 1)
            thf = np.median(mvf) + threshold
            f = np.fft.fftfreq(noDP, dt)
            f = np.fft.fftshift(f)
            ranges = np.linspace(0, noRG - 1, noRG)
            location = np.array(np.where(powerdb > thf))
            mask = np.where(powerdb > thf, 1, 0)
            dataf = powerdb * mask
            slices = find_objects(dataf)
            nonsf = list()
            for l in range(np.size(slices, 0)):
                nonsf.append([slices[l][1].start, slices[l][1].stop, slices[l][0].start, slices[l][0].stop])
            fig = plt.figure(figsize=(25, 9))
            ax = fig.add_subplot(111)
            pp = ax.imshow(powerdb, extent=[min(f), max(f), min(ranges), max(ranges)], origin='lower',
                            aspect="auto", cmap="inferno")
            plt.colorbar(pp)
            plt.title('Frequency Domain/Detected events')
            plt.xlabel('Frequency/Hz')
            plt.ylabel('Ranges')
            plt.show()



def sub(source_dir, subfolder_size):
    if not os.path.exists(source_dir):
        print("Source directory doesn't exist.")
        return

    if subfolder_size <= 0:
        print("Subfolder size should be greater than zero.")
        return

    file_list = os.listdir(source_dir)
    total_files = len(file_list)
    num_subfolders = ceil(total_files / subfolder_size)

    if total_files == 0:
        print("No files to organize.")
        return

    for i in range(num_subfolders):
        subfolder_name = f"time{i}"
        subfolder_path = os.path.join(source_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        for j in range(i * subfolder_size, min((i + 1) * subfolder_size, total_files)):
            source_file = file_list[j]
            source_path = os.path.join(source_dir, source_file)
            target_path = os.path.join(subfolder_path, source_file)
            shutil.move(source_path, target_path)
            print(f"Moved {source_file} to {subfolder_name}")

                
