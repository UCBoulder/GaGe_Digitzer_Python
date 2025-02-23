"""
All the PyGage commands sends a command to the digitizer, the digitizer will
send back a success or error signal. If the command was a query it will either
return the requested information, or an error signal.
"""
import sys

sys.path.append("include")
import matplotlib.pyplot as plt
import gc
from configparser import ConfigParser  # needed to read ini files
import threading
import time
import PyGage3_64 as PyGage
import GageSupport as gs
import numpy as np


from GageConstants import (
    CS_CURRENT_CONFIGURATION,
    CS_ACQUISITION_CONFIGURATION,
    CS_STREAM_TOTALDATA_SIZE_BYTES,
    CS_DATAPACKING_MODE,
    CS_MASKED_MODE,
    CS_GET_DATAFORMAT_INFO,
    CS_BBOPTIONS_STREAM,
    CS_MODE_USER1,
    CS_MODE_USER2,
    CS_EXTENDED_BOARD_OPTIONS,
    STM_TRANSFER_ERROR_FIFOFULL,
    CS_SEGMENTTAIL_SIZE_BYTES,
    CS_TIMESTAMP_TICKFREQUENCY,
)
from GageErrors import (
    CS_MISC_ERROR,
    CS_INVALID_PARAMS_ID,
    CS_STM_TRANSFER_TIMEOUT,
    CS_STM_COMPLETED,
)
import array

# default parameters
TRANSFER_TIMEOUT = -1  # milliseconds
STREAM_BUFFERSIZE = 0x200000  # 2097152
MAX_SEGMENT_COUNT = 25000
inifile_default = "include/Stream2Analysis_CARD1.ini"
inifile_acquire_default = "include/Acquire_CARD1.ini"


# class used to hold streaming information, attributes listed in __slots__
# below are assigned later in the card_stream function. The purpose of
# __slots__ is that it does not allow the user to add new attributes not listed
# in __slots__ (a kind of neat implementation I hadn't known about).
class StreamInfo:
    __slots__ = [
        "WorkBuffer",
        "TimeStamp",
        "BufferSize",
        "SegmentSize",
        "TailSize",
        "LeftOverSize",
        "BytesToEndSegment",
        "BytesToEndTail",
        "DeltaTime",
        "LastTimeStamp",
        "Segment",
        "SegmentCountDown",
        "SplitTail",
    ]


# finding a gauge card and returning the handle
# the handle is just an integer used to identify the card,
# and sort of "is" the gage card
def get_handle():
    status = PyGage.Initialize()
    if status < 0:
        return status
    else:
        handle = PyGage.GetSystem(0, 0, 0, 0)
        return handle


def configure_system(handle, inifile):
    acq, sts = gs.LoadAcquisitionConfiguration(handle, inifile)

    if isinstance(acq, dict) and acq:
        status = PyGage.SetAcquisitionConfig(handle, acq)
        if status < 0:
            return status
    else:
        print("Using defaults for acquisition parameters")
        status = PyGage.SetAcquisitionConfig(handle, acq)

    if sts == gs.INI_FILE_MISSING:
        print("Missing ini file, using defaults")
    elif sts == gs.PARAMETERS_MISSING:
        print(
            "One or more acquisition parameters missing, "
            "using defaults for missing values"
        )

    system_info = PyGage.GetSystemInfo(handle)

    if not isinstance(
        system_info, dict
    ):  # if it's not a dict, it's an int indicating an error
        return system_info

    channel_increment = gs.CalculateChannelIndexIncrement(
        acq["Mode"],
        system_info["ChannelCount"],
        system_info["BoardCount"],
    )

    missing_parameters = False
    for i in range(1, system_info["ChannelCount"] + 1, channel_increment):
        chan, sts = gs.LoadChannelConfiguration(handle, i, inifile)
        if isinstance(chan, dict) and chan:
            status = PyGage.SetChannelConfig(handle, i, chan)
            if status < 0:
                return status
        else:
            print("Using default parameters for channel ", i)
        if sts == gs.PARAMETERS_MISSING:
            missing_parameters = True

    if missing_parameters:
        print(
            "One or more channel parameters missing, "
            "using defaults for missing values"
        )

    missing_parameters = False
    # in this example we're only using 1 trigger source, if we use
    # system_info['TriggerMachineCount'] we'll get warnings about
    # using default values for the trigger engines that aren't in
    # the ini file
    trigger_count = 1
    for i in range(1, trigger_count + 1):
        trig, sts = gs.LoadTriggerConfiguration(handle, i, inifile)
        if isinstance(trig, dict) and trig:
            status = PyGage.SetTriggerConfig(handle, i, trig)
            if status < 0:
                return status
        else:
            print("Using default parameters for trigger ", i)

        if sts == gs.PARAMETERS_MISSING:
            missing_parameters = True

    if missing_parameters:
        print(
            "One or more trigger parameters missing, "
            "using defaults for missing values"
        )

    g_cardTotalData = []
    g_segmentCounted = []
    for i in range(system_info["BoardCount"]):
        g_cardTotalData.append(0)
        g_segmentCounted.append(0)

    return status, g_cardTotalData, g_segmentCounted


def load_stm_configuration(inifile):
    app = {}
    # set reasonable defaults

    app["TimeoutOnTransfer"] = TRANSFER_TIMEOUT
    app["BufferSize"] = STREAM_BUFFERSIZE
    app["DoAnalysis"] = 0
    app["ResultsFile"] = "Result"

    config = ConfigParser()

    # parse existing file
    config.read(inifile)
    section = "StmConfig"

    if section in config:
        for key in config[section]:
            key = key.lower()
            value = config.get(section, key)
            if key == "doanalysis":
                if int(value) == 0:
                    app["DoAnalysis"] = False
                else:
                    app["DoAnalysis"] = True
            elif key == "timeoutontransfer":
                app["TimeoutOnTransfer"] = int(value)
            elif key == "buffersize":  # in bytes
                app["BufferSize"] = int(value)  # may need to be an int64
            elif key == "resultsfile":
                app["ResultsFile"] = value
    return app


def check_for_expert_stream(handle):
    expert_options = CS_BBOPTIONS_STREAM

    # get the big acquisition configuration dict
    acq = PyGage.GetAcquisitionConfig(handle)

    if not isinstance(acq, dict):
        if not acq:
            print("Error in call to GetAcquisitionConfig")
            return CS_MISC_ERROR
        else:  # should be error code
            print("Error: ", PyGage.GetErrorString(acq))
            return acq

    extended_options = PyGage.GetExtendedBoardOptions(handle)
    if extended_options < 0:
        print("Error: ", PyGage.GetErrorString(extended_options))
        return extended_options

    if extended_options & expert_options:
        print("\nSelecting Expert Stream from image 1")
        acq["Mode"] |= CS_MODE_USER1
    elif (extended_options >> 32) & expert_options:
        print("\nSelecting Expert Stream from image 2")
        acq["Mode"] |= CS_MODE_USER2

    # I'm getting an unknown error signal, so comment this out:

    # comment out -------------------------------------------------------------
    # else:
    #     print("\nCurrent system does not support Expert Streaming")
    #     print("\nApplication terminated")
    #     return CS_MISC_ERROR
    # -------------------------------------------------------------------------

    else:
        # the eXpert image is loaded on Image1
        acq["Mode"] |= CS_MODE_USER1
        pass

    status = PyGage.SetAcquisitionConfig(handle, acq)
    if status < 0:
        print("Error: ", PyGage.GetErrorString(status))
    return status


def initialization_before_streaming(inifile, buffer_size):
    handle = get_handle()
    if handle < 0:
        # get error string
        error_string = PyGage.GetErrorString(handle)
        print("Error: ", error_string)

        return

    system_info = PyGage.GetSystemInfo(handle)
    if not isinstance(
        system_info, dict
    ):  # if it's not a dict, it's an int indicating an error
        error_string = PyGage.GetErrorString(system_info)
        print("Error: ", error_string)
        PyGage.FreeSystem(handle)

        return
    print("\nBoard Name: ", system_info["BoardName"])

    # get streaming parameters
    app = load_stm_configuration(inifile)

    # configure system
    status, g_cardTotalData, g_segmentCounted = configure_system(handle, inifile)
    if status < 0:
        # get error string
        error_string = PyGage.GetErrorString(status)
        print("Error: ", error_string)
        PyGage.FreeSystem(handle)

        return

    # initialize the stream
    status = check_for_expert_stream(handle)
    if status < 0:
        # error string is printed out in check_for_expert_stream
        PyGage.FreeSystem(handle)
        raise SystemExit

    # This function sends configuration parameters that are in the driver
    # to the CompuScope system associated with the handle. The parameters
    # are sent to the driver via SetAcquisitionConfig. SetChannelConfig and
    # SetTriggerConfig. The call to Commit sends these values to the
    # hardware. If successful, the function returns CS_SUCCESS (1).
    # Otherwise, a negative integer representing a CompuScope error code is
    # returned.
    status = PyGage.Commit(handle)
    if status < 0:
        # get error string
        error_string = PyGage.GetErrorString(status)
        print("Error: ", error_string)
        PyGage.FreeSystem(handle)

        return
        # raise SystemExit

    # -------------------------------------------------------------------------
    # initialization done

    if buffer_size is not None:
        app["BufferSize"] = buffer_size

    return handle, app, g_cardTotalData, g_segmentCounted, system_info


# %% ===== preparation ========================================================
(
    handle,
    app,
    g_cardTotalData,
    g_segmentCounted,
    system_info,
) = initialization_before_streaming(inifile_default, int(100e6))

# Returns the frequency of the timestamp counter in Hertz for the CompuScope system associated with the
# handle. negative if an error occurred
g_tickFrequency = PyGage.GetTimeStampFrequency(handle)
if g_tickFrequency < 0:
    print("Error: ", PyGage.GetErrorString(g_tickFrequency))
    PyGage.FreeSystem(handle)
    raise SystemExit

# after commit the sample size may change
# get the big acquisition configuration dict
acq_config = PyGage.GetAcquisitionConfig(handle)

# get total amount of data we expect to receive in bytes, negative if an error occurred
total_samples = PyGage.GetStreamTotalDataSizeInBytes(handle)

if total_samples < 0 and total_samples != acq_config["SegmentSize"]:
    print("Error: ", PyGage.GetErrorString(total_samples))
    PyGage.FreeSystem(handle)
    raise SystemExit

# convert from bytes -> samples and print it to screen
if total_samples != -1:
    total_samples = total_samples // system_info["SampleSize"]
    print("total samples is: ", total_samples)

card_index = 1
buffer1 = PyGage.GetStreamingBuffer(handle, card_index, app["BufferSize"])
if isinstance(buffer1, int):
    PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
    print("Error getting streaming buffer 1: ", PyGage.GetErrorString(buffer1))

buffer2 = PyGage.GetStreamingBuffer(handle, card_index, app["BufferSize"])
if isinstance(buffer2, int):
    print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer2))
    PyGage.FreeStreamingBuffer(handle, card_index, buffer2)

# number of samples in data segment
acq = PyGage.GetAcquisitionConfig(handle)
data_in_segment_samples = acq["SegmentSize"] * (acq["Mode"] & CS_MASKED_MODE)

status = PyGage.GetSegmentTailSizeInBytes(handle)
if status < 0:
    print("Error: ", PyGage.GetErrorString(status))
segment_tail_size_in_bytes = status
tail_left_over = 0

sample_size = system_info["SampleSize"]
segment_size_in_bytes = data_in_segment_samples * sample_size
transfer_size_in_samples = app["BufferSize"] // sample_size
print("\nActual buffer size used for data streaming = ", app["BufferSize"])
print("\nActual sample size used for data streaming = ", transfer_size_in_samples)

stream_info = StreamInfo()
stream_info.WorkBuffer = np.zeros_like(buffer1)
stream_info.TimeStamp = array.array("q")
stream_info.BufferSize = app["BufferSize"]
stream_info.SegmentSize = segment_size_in_bytes
stream_info.TailSize = segment_tail_size_in_bytes
stream_info.BytesToEndSegment = segment_size_in_bytes
stream_info.BytesToEndTail = segment_tail_size_in_bytes
stream_info.LeftOverSize = tail_left_over
stream_info.LastTimeStamp = 0
stream_info.Segment = 1
stream_info.SegmentCountDown = acq["SegmentCount"]
stream_info.SplitTail = False

# %% ===== stream =============================================================
done = False
stream_completed_success = False
loop_count = 0
work_buffer_active = False

# start the capture!
status = PyGage.StartCapture(handle)
if status < 0:
    # get error string
    print("Error: ", PyGage.GetErrorString(status))
    PyGage.FreeSystem(handle)
    raise SystemExit

while not done and not stream_completed_success:
    # select which buffer to stream 2 (toggled in each loop count)
    if loop_count & 1:
        buffer = buffer2

    else:
        buffer = buffer1

    # start transfer
    status = PyGage.TransferStreamingData(
        handle, card_index, buffer, transfer_size_in_samples
    )
    if status < 0:
        if status == CS_STM_COMPLETED:
            # pass (-803 just indicates that the streaming acquisition
            # completed)
            stream_completed = True
        else:
            print("Error: ", PyGage.GetErrorString(status))
            break

    # query for transfer status
    p = PyGage.GetStreamingTransferStatus(handle, card_index, app["TimeoutOnTransfer"])
    if isinstance(p, tuple):
        g_cardTotalData[0] += p[1]  # have total_data be an array, 1 for each card
        if p[2] == 0:
            stream_completed_success = False
        else:
            stream_completed_success = True

        if STM_TRANSFER_ERROR_FIFOFULL & p[0]:
            print("Fifo full detected on card ", card_index)
            done = True

    else:  # error detected
        done = True
        if p == CS_STM_TRANSFER_TIMEOUT:
            print("\nStream transfer timeout on card ", card_index)
        else:
            print("5 Error: ", p)
            print("5 Error: ", PyGage.GetErrorString(p))

    # set workbuffer to the buffer that the new data was just transferred into
    stream_info.WorkBuffer[:] = buffer
    work_buffer_active = True
    loop_count += 1

    if loop_count % 10 == 0:
        print(loop_count)

PyGage.FreeSystem(handle)
