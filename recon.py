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
    title_text = f'{histogram_name} Histogram\nMost Repeated Value: {most_repeated_value:.2f}, Median: {median_value:.2f}'
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
            mvf = np.median(SNRf, 1)# Calculate the mean power in each range gate
            NLl = np.median(mvf)#noise level in linear scale
            th=3+NLl
            mask = np.where(SNRf[50:120,1000:2000]>th, 1, 0)# Create a binary mask to mark the detected locations
            slices=find_objects(mask)
            PMSE=list()
            for l in range(np.size(slices,0)):
                PMSE.append([slices[l][1].start+1000,slices[l][1].stop+1000,slices[l][0].start+int(ranges[50]),slices[l][0].stop+int(ranges[50])])
            PMSE=combine_inboxes(PMSE)#combining the boxes included inside each other together
            F_threshold=10
            range_threshold=9
            PMSE=filter_boxes(PMSE, F_threshold, range_threshold)#removing small boxes
            largest4,largest =biggest_boxes(PMSE,noDP)#finding larges 3 boxes and combining them together
            if largest==[1500,1500,0,0]:
                print('no PMSE detected.')
            if largest!=[1500,1500,0,0]:
                print("PMSE detected")
                print("removing echoes other than PMSE")
                time=np.zeros(data.shape, dtype=complex)
                for i in range (noRX):
                    for j in range (noTX):
                        for k in range (noRG):
                            ti,time[i,j,k,:]=make_ifft(f,data[i,j,k,:],dt)

                powert =np.mean(np.abs(time)**2,axis=(0,1))# Calculate the power spectrum
                powerdbt = 10*np.log10(powert)# Convert the power spectrum to dB sca
                mvf = np.median(powert, 1)# Calculate the mean power in each range gate
                NLl = np.median(mvf)#noise level in linear scale
                NL = 10 * np.log10(NLl)  # Convert NL to dB
                SNRt = 10 * np.log10(powert/NLl)# Calculate SNR in dB
        # =============================================================================
        # removing DC
        # =============================================================================
                mvf = np.median(powert, 1)# Calculate the mean power in each range gate
                NLl = np.median(mvf)#noise level in linear scale
                NL = 10 * np.log10(NLl)  # Convert NL to dB
                SNRtt = 10 * np.log10(powert/NLl)# JUST FOR DETECTION
                lowsnr=SNRtt[0:50,:].copy()
                SNRtt=SNRtt+abs(np.min(SNRtt))
                mf = np.mean(SNRtt, 1)#removing DC
                # Reshape mf to (350, 1) to make the shapes compatible
                mf_reshaped = mf[:, np.newaxis]
                SNRtt = SNRtt/mf_reshaped
        # =============================================================================
        #     detection
        # =============================================================================
                th=1.25
                mask = np.where(SNRtt>th, 1, 0)# Create a binary mask to mark the detected locations
                dataf=SNRtt*mask
                slices=find_objects(mask)
                nonsf=list()
                for l in range(np.size(slices,0)):
                    nonsf.append([slices[l][1].start,slices[l][1].stop,slices[l][0].start,slices[l][0].stop])
        # =============================================================================
        # replacing with noise
        # =============================================================================
                mvf = np.median(powert, 1)# Calculate the mean power in each range gate
                NLl = np.median(mvf)#noise level
                for i in range(np.size(nonsf,0)):
                    x_start, x_end, y_start, y_end =nonsf[i][0],nonsf[i][1],nonsf[i][2],nonsf[i][3]
                    for j in range(y_start,y_end):
                        for k in range(noRX):
                            for l in range(noTX):
                                np.put(time[k,l,j,:],np.arange(x_start,x_end),np.sqrt(mvf[j]/2)*(np.random.randn(x_end-x_start)+1j*np.random.randn(x_end-x_start)))
                                
                fft = np.zeros(time.shape, dtype=complex)# Initialize an array 'fft' to store FFT results
                for i in range(noRX):
                    for j in range(noTX):
                        for k in range (noRG):
                            #time[i,j,k,:]=smooth_signal(time[i,j,k,:],51)
                            fi,fft[i,j,k,:]=make_fft(ti,time[i,j,k,:])
                            fft[i,j,k,:]=smooth(fft[i,j,k,:],30)
                powert2 =np.mean(np.abs(time)**2,axis=(0,1))
                powerdbt2 = 10*np.log10(powert2)# Convert the power spectrum to dB sca
                mvf = np.median(powert2, 1)# Calculate the mean power in each range gate
                NLl = np.median(mvf)#noise level in linear scale
                NL = 10 * np.log10(NLl)  # Convert NL to dB
                SNRt2 = 10 * np.log10(powert2/NLl)# Calculate SNR in dB
                
                powerf2 = np.mean(np.abs(fft) ** 2, axis=(0, 1))
                powerdbf2 = 10 * np.log10(powerf2)
                mvf = np.median(powerf2, 1)# Calculate the mean power in each range gate
                NLl = np.median(mvf)#noise level in linear scale
                NL = 10 * np.log10(NLl)  # Convert NL to dB
                SNRf2 = 10 * np.log10(powerf2/NLl)# Calculate SNR in dB
                #ranges = ranges * 3 / 2

        # =============================================================================
        #         plotting
        # =============================================================================
                #cc='viridis'#"inferno"#
                # cc = "plasma"
                # cc = "inferno"
                # cc = "cividis"
                cc = "jet"
                #cc="inferno"#
                ranges=ranges*3/2
                vminf,vmaxf,vmint,vmaxt=np.min(SNRf),np.max(SNRf),np.min(SNRt),np.max(SNRt)
                #vminf,vmaxf,vmint,vmaxt=3,20,0,20
                vmaxf,vmaxt=20,20
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(18, 8))
                dt_object = datetime.utcfromtimestamp(currenttime)
                fig.suptitle(f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                im1 = ax1.imshow(SNRf, extent=[min(f), max(f), min(ranges), max(ranges)], origin='lower',
                        aspect="auto", cmap=cc,vmin=vminf,vmax=vmaxf)
                ax1.set_title('SNR')
                ax1.set_xlabel('Frequency/Hz')
                ax1.set_ylabel('Range/Km')
                plt.colorbar(im1, ax=ax1,label='SNR/dB.')
                im2 = ax2.imshow(SNRt, extent=[min(ti), max(ti), min(ranges), max(ranges)], origin='lower',
                                aspect="auto", cmap=cc,vmin=vmint,vmax=vmaxt)#,vmin=-30)
                ax2.set_title('SNR')
                ax2.set_xlabel('Time/Sec')
                ax2.set_ylabel('Range/Km')
                plt.colorbar(im2, ax=ax2,label='SNR/dB.')

                plt.tight_layout()  # To ensure that the plots don't overlap
                im3 = ax3.imshow(SNRf2, extent=[min(f), max(f), min(ranges), max(ranges)], origin='lower',
                            aspect="auto", cmap=cc,vmin=vminf,vmax=vmaxf)
                ax3.set_title('SNR')
                ax3.set_xlabel('Frequency/Hz')
                ax3.set_ylabel('Range/Km')
                plt.colorbar(im3, ax=ax3,label='SNR/dB.')
                im4 = ax4.imshow(SNRt2, extent=[min(ti), max(ti), min(ranges), max(ranges)], origin='lower',
                                aspect="auto", cmap=cc,vmin=vmint,vmax=vmaxt)#,vmin=-30)
                ax4.set_title('SNR')
                ax4.set_xlabel('Time/Sec')
                ax4.set_ylabel('Range/Km')
                plt.colorbar(im4, ax=ax4,label='SNR/dB.')
                plt.tight_layout()
                plt.show()
                
          # =============================================================================
          # statistics
          # =============================================================================   
                print("extracting statistics")
                ranges = np.linspace(0, noRG - 1, noRG)
                #SNRf2=smooth(SNRf2, sigma=6)
                mvf = np.median(SNRf2, 1)# Calculate the mean power in each range gate
                NLl = np.median(mvf)#noise level in lDB
                th=NLl+3
                mask = np.where(SNRf2[50:120,1000:2000]>th, 1, 0)# Create a binary mask to mark the detected locations
                slices=find_objects(mask)
                PMSE=list()
                for l in range(np.size(slices,0)):
                    PMSE.append([slices[l][1].start+1000,slices[l][1].stop+1000,slices[l][0].start+int(ranges[50]),slices[l][0].stop+int(ranges[50])])
                PMSE=combine_inboxes(PMSE)#combining the boxes included inside each other together
                F_threshold=10
                range_threshold=9
                PMSE=filter_boxes(PMSE, F_threshold, range_threshold)#removing small boxes
                largest4,largest =biggest_boxes(PMSE,noDP)#finding larges 3 boxes and combining them together
                powerprofilesec=np.sum(powerf2[largest[2]:largest[3],largest[0]:largest[1]], 1)
        # =============================================================================
        #         plotting
        # =============================================================================
                print(PMSE)
                #vminf, vmaxf = 3, 20
                ranges=ranges*3/2
                fmin,fmax,Amin,Amax="{:.2f}".format(f[largest[0]]),"{:.2f}".format(f[largest[1]]),int(ranges[largest[2]]),int(ranges[largest[3]])
                NLl2,th2="{:.2f}".format(NLl),"{:.2f}".format(th)

                datev,Timev=radar_date(currenttime)
                if largest!=[1500,1500,0,0]:
                    fig, ((ax1, ax2)) = plt.subplots(1, 2,figsize=(18, 8))
                    dt_object = datetime.utcfromtimestamp(currenttime)
                    fig.suptitle(f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    im1 = ax1.imshow(SNRf2, extent=[min(f), max(f), min(ranges), max(ranges)], origin='lower',
                                        aspect="auto", cmap=cc, vmin=vminf, vmax=vmaxf)
                    ax1.add_patch(Rectangle((f[largest[0]], ranges[largest[2]]),
                                                f[largest[1]-1] - f[largest[0]], (ranges[largest[3]-1] - ranges[largest[2]]),
                                                fc='none', ec='r', lw=2))
                    ax1.set_title('SNR')
                    ax1.set_xlabel('Frequency/Hz')
                    ax1.set_ylabel('Range/Km')
                    plt.colorbar(im1, ax=ax1,label='SNR/dB')
                    im2 = ax2.imshow(SNRf2, extent=[min(f), max(f), min(ranges), max(ranges)], origin='lower',
                                        aspect="auto", cmap=cc, vmin=vminf, vmax=vmaxf)
                    ax2.set_xlim([f[largest[0]], f[largest[1]-1]])
                    ax2.set_ylim([ranges[largest[2]], ranges[largest[3]-1]])
                    text_value = f"PMSE statistics\nFmin: {fmin}Hz\nF max: {fmax}Hz\nStart range: {Amin}Km\nEnd range: {Amax}Km\nNoise level:{NLl2} dB\nThreshold:{th2} dB"
                    ax2.text(0.95, 0.95, text_value, color='red', ha='right', va='top', transform=ax2.transAxes,
                                bbox=dict(facecolor='white', alpha=0.7))
                    ax2.set_title('SNR')
                    ax2.set_xlabel('Frequency/Hz')
                    ax2.set_ylabel('Range/Km')
                    cbar = plt.colorbar(im2, ax=ax2)
                    cbar.set_label('SNR/dB')
                    plt.tight_layout()
                    plt.show()
                    fmin_values.append(float(f[largest[0]]))
                    fmax_values.append(float(f[largest[1]]))
                    Amin_values.append(int(ranges[largest[2]]))
                    Amax_values.append(int(ranges[largest[3]]))
                    time_values.append(float(currenttime))
                    print("Time:",Timev)            
                    print("Date:",datev)
                    print("start Range:",int(ranges[largest[2]]),"Km")
                    print("End Range:",int(ranges[largest[3]]),"Km")
                    print("start Frequency:",float(f[largest[0]]),"Hz")
                    print("end Frequency:",float(f[largest[1]]),"Hz")
                    pmsef=f[largest[0]:largest[1]]
                    pmseranges=ranges[largest[2]:largest[3]]
                    pmseSNR=SNRf2[largest[2]:largest[3],largest[0]:largest[1]]
                    ci=5
                    pmseSNR2=list()
                    for i in range(np.size(pmseranges)):
                        pmseSNR3=make_ci(pmsef,pmseSNR[i], ci)
                        pmseSNR2.append(pmseSNR3)
                    pmseSNR2=np.array(pmseSNR2) 
                    min_value=min(pmsef)
                    max_value=max(pmsef)
                    vector_length=len(pmseSNR2[0])
                    pmsef2= np.linspace(min_value, max_value, vector_length)
                    detailed_statistics=list()
                    for i in range(np.size(pmseSNR2,0)):
                        indexmin,indexmax=find_fminmax(pmseSNR2[i,:],th,3)
                        fmin=pmsef2[indexmin]
                        fmax=pmsef2[indexmax]
                        if indexmin!=-1 and indexmax!=-1:
                            detailed_statistics.append([fmin,fmax,pmseranges[i]])
                    detailed_statistics=np.array(detailed_statistics)
                    col_namesg =["F min/Hz", "F max/Hz","Range/Km"]
                    ds_dtg = np.dtype( { 'names':col_namesg,'formats':[ (float), (float), (float)] } )
                    detailed_statistics= np.rec.array(detailed_statistics,dtype=ds_dtg)
                    main_file_path = os.path.join("statistics","detailed_statistics.h5")
                    # Check if the main HDF5 file exists
                    if not os.path.exists(main_file_path):
                        # Create a new HDF5 file
                        with h5py.File(main_file_path, 'w') as main_file:
                            main_file.create_dataset(f"{currenttime}", data=detailed_statistics)
                    else:
                        with h5py.File(main_file_path, 'a') as main_file:
                            main_file.create_dataset(f"{currenttime}", data=detailed_statistics)
                    
# =============================================================================
#                     calculating the power profile noise
# =============================================================================
                    powerprofilenoise=np.median(np.median(powerf2, 1))*np.size(pmseSNR,1)
                    main_file_path = os.path.join("power_profile","detailed_PMSE_power_profile.h5")
                    # Check if the main HDF5 file exists
                    if not os.path.exists(main_file_path):
                        # Create a new HDF5 file
                        with h5py.File(main_file_path, 'w') as main_file:
                            group = main_file.create_group(f"{currenttime}")
                            group.create_dataset("power_profile_data", data=powerprofilesec)
                            group.create_dataset("power_profile_noise", data=powerprofilenoise)
                    else:
                        with h5py.File(main_file_path, 'a') as main_file:
                            group = main_file.create_group(f"{currenttime}")
                            group.create_dataset("power_profile_data", data=powerprofilesec)
                            group.create_dataset("power_profile_noise", data=powerprofilenoise)
    
    print("calculating the power profile for all of the processed data")                
    powerpr = np.array(powerpr)
    powerpr= powerpr.T
    mins = int(np.size(powerpr, 1) / 2)
    hours=mins/60
    timePR = np.linspace(0,hours, np.size(powerpr, 1))
    powerpr = powerpr / 10000
    mf = np.mean(powerpr, 0)
    powerpr = powerpr - mf
    power_path = os.path.join(os.getcwd(),'power_profile','power_profile_picture.png')
    data_path=os.path.join(os.getcwd(),'power_profile','power_profile_data.npy')
    plt.figure(figsize=(25, 9), dpi=600)
    pp = plt.imshow(powerpr, extent=[min(timePR), max(timePR), min(ranges), max(ranges)],
                    origin='lower', aspect="auto", cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(pp)
    plt.title('power_profile')
    plt.xlabel('Time (hours)')
    plt.ylabel('Ranges')
    plt.savefig(power_path, dpi=1000,bbox_inches='tight')
    plt.show()

    # Check if the directory already exists
    hdf5_file_path = os.path.join("power_profile","all_day_power_profile_data")
    with h5py.File(hdf5_file_path, "w") as f5:
        f5.create_dataset("PowerProfile", data=powerpr)
        f5.create_dataset("Time", data=timePR)
        f5.create_dataset("Ranges", data=ranges)
        

    # =============================================================================
    # histogram for the statistics
    # =============================================================================
    # Set Seaborn style
    statistics = combine_lists(fmin_values,fmax_values,Amin_values,Amax_values,time_values)
    col_namesg =["F min/Hz", "F max/Hz","Range Min/Km","Range Max/Km","Time"]
    ds_dtg = np.dtype( { 'names':col_namesg,'formats':[ (float), (float), (float), (float), (float)] } )
    statistics= np.rec.array(statistics,dtype=ds_dtg)
    hdf5_file_path = os.path.join("statistics","General_statistics")
    with h5py.File(hdf5_file_path, "w") as f6:
        col_names = ["F min/HZ", "F max/HZ","Range Min/Km,Range Max/Km","Time"]
        ds_dt = np.dtype({'names': col_names, 'formats': [float, float,float, float,float, float]})
        Statistics = f6.create_dataset("statistics", data=statistics)

    sns.set(style="whitegrid")
    histo_path = os.path.join("statistics","statistics_histogram.png")
    # Draw histogram
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), dpi=100)
    highlight_most_repeated(axs[0, 0], fmin_values, sns.color_palette("Blues")[4], 'Fmin/Hz', 'Frequency/Hz', 'Range')
    highlight_most_repeated(axs[0, 1], fmax_values, sns.color_palette("Oranges")[4], 'Fmax/Hz', 'Frequency/Hz', 'Range')
    highlight_most_repeated(axs[1, 0], Amin_values, sns.color_palette("Greens")[4], 'Range min/Km', 'Range/Km', 'Values')
    highlight_most_repeated(axs[1, 1], Amax_values, sns.color_palette("Reds")[4], 'Range max/Km', 'Range/Km', 'Values')
    # Adjust legend font size and add a box around it
    for ax in axs.flat:
        leg = ax.legend()
        if leg:
            leg.get_frame().set_linewidth(1.5)
            leg.get_frame().set_edgecolor('black')
            for text in leg.get_texts():
                text.set_fontsize('medium')

    # Adjust figure layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(histo_path, dpi=1000,bbox_inches='tight')
    plt.show()
    
    
data_directory =os.path.join(os.getcwd(),'data')
rebuild_statistics(data_directory)
