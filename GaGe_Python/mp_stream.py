"""
All the PyGage commands sends a command to the digitizer, the digitizer will
send back a success or error signal. If the command was a query it will either
return the requested information, or an error signal.
"""
import matplotlib.pyplot as plt
import gc
from configparser import ConfigParser  # needed to read ini files
import threading
import time
import PyGage3_64 as PyGage
import GageSupport as gs
import numpy as np
import multiprocessing as mp


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
inifile_default = "Stream2Analysis.ini"
inifile_acquire_default = "Acquire.ini"


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

    else:
        print("\nCurrent system does not support Expert Streaming")
        print("\nApplication terminated")
        return CS_MISC_ERROR

    status = PyGage.SetAcquisitionConfig(handle, acq)
    if status < 0:
        print("Error: ", PyGage.GetErrorString(status))
    return status


def initialize_stream(inifile, buffersize):
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

    if buffersize is not None:
        app["BufferSize"] = buffersize

    return handle, app, g_cardTotalData, g_segmentCounted, system_info


def stream(
    inifile,
    buffersize,
    stream_ready_event,
    stream_start_event,
    stream_stop_event,
    stream_error_event,
    N_threads=2,
    mp_values=[],
    mp_arrays=[],
    args_doanalysis=None,
):
    """
    GaGe card streaming. This is a process independent function that can be
    passed to mp.Process

    Args:
        inifile (string): path to inifle
        buffersize (int): streaming buffer size to allocate
        stream_ready_event (mp.Event): stream is ready, set by this function
        stream_start_event (mp.Event): stream started, set by this function
        stream_stop_event (mp.Event): stream stopped by user, checked by this function
        stream_error_event (mp.Event): stream error, set by this function
        N_threads (int, optional): Description
        mp_values (list, optional): list of mp.Value's for real-time analysis
        mp_arrays (list, optional): list of mp.Array's for real-time analysis
        args_doanalysis (None, optional): any additional arguments to pass for real-time analysis
    """
    # %% ====== handle and config =============================================
    (
        handle,
        app,
        g_cardTotalData,
        g_segmentCounted,
        system_info,
    ) = initialize_stream(inifile, buffersize)

    # Returns the frequency of the timestamp counter in Hertz for the
    # CompuScope system associated with the handle. negative if an error
    # occurred
    g_tickFrequency = PyGage.GetTimeStampFrequency(handle)
    if g_tickFrequency < 0:
        print("Error: ", PyGage.GetErrorString(g_tickFrequency))
        PyGage.FreeSystem(handle)
        raise SystemExit

    # after commit the sample size may change
    # get the big acquisition configuration dict
    acq_config = PyGage.GetAcquisitionConfig(handle)

    # get total amount of data we expect to receive in bytes, negative if an
    # error occurred
    total_samples = PyGage.GetStreamTotalDataSizeInBytes(handle)

    if total_samples < 0 and total_samples != acq_config["SegmentSize"]:
        print("Error: ", PyGage.GetErrorString(total_samples))
        PyGage.FreeSystem(handle)
        raise SystemExit

    # convert from bytes -> samples and print it to screen
    if total_samples != -1:
        total_samples = total_samples // system_info["SampleSize"]
        print("total samples is: ", total_samples)

    # ======================== initalize streaming buffers ====================
    card_index = 1
    buffer1 = PyGage.GetStreamingBuffer(handle, card_index, app["BufferSize"])
    if isinstance(buffer1, int):
        print("Error getting streaming buffer 1: ", PyGage.GetErrorString(buffer1))
    buffer2 = PyGage.GetStreamingBuffer(handle, card_index, app["BufferSize"])
    if isinstance(buffer2, int):
        print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer2))
        PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
    buffer3 = PyGage.GetStreamingBuffer(handle, card_index, app["BufferSize"])
    if isinstance(buffer3, int):
        print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer3))
        PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
        PyGage.FreeStreamingBuffer(handle, card_index, buffer2)
    buffer4 = PyGage.GetStreamingBuffer(handle, card_index, app["BufferSize"])
    if isinstance(buffer4, int):
        print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer4))
        PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
        PyGage.FreeStreamingBuffer(handle, card_index, buffer2)
        PyGage.FreeStreamingBuffer(handle, card_index, buffer3)

    buffer_list = [buffer1, buffer2, buffer3, buffer4]

    #  =========== stream_info instance =======================================
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

    # %% ========== stream ====================================================
    done = False
    stream_completed_success = False
    work_buffer_active = False
    loop_count = 0
    buffer_count = 0
    thread_count = 0

    # thread count
    work_threads = [None for i in range(N_threads)]
    active_threads = [False for i in range(N_threads)]
    print(f"using {len(work_threads)} threads for analysis")

    stream_ready_event.set()

    # start the capture!
    status = PyGage.StartCapture(handle)
    if status < 0:
        # get error string
        print("Error: ", PyGage.GetErrorString(status))
        PyGage.FreeSystem(handle)
        raise SystemExit

    stream_start_event.set()

    while not done and not stream_completed_success:
        # select which buffer to stream 2 (toggled in each loop count)
        buffer = buffer_list[buffer_count]

        # ===== start transfer ================================================
        status = PyGage.TransferStreamingData(
            handle, card_index, buffer, transfer_size_in_samples
        )
        if status < 0:
            if status == CS_STM_COMPLETED:
                # pass (-803 just indicates that the streaming acquisition
                # completed)
                pass
            else:
                print("Error: ", PyGage.GetErrorString(status))
                break

        # ===== query for transfer status =====================================
        p = PyGage.GetStreamingTransferStatus(
            handle, card_index, app["TimeoutOnTransfer"]
        )
        if isinstance(p, tuple):
            g_cardTotalData[0] += p[1]  # have total_data be an array, 1 for each card
            if p[2] == 0:
                stream_completed_success = False
            else:
                stream_completed_success = True

            if STM_TRANSFER_ERROR_FIFOFULL & p[0]:
                print("Fifo full detected on card ", card_index)
                done = True
                stream_error_event.set()

        else:  # error detected
            done = True
            stream_error_event.set()
            if p == CS_STM_TRANSFER_TIMEOUT:
                print("\nStream transfer timeout on card ", card_index)
            else:
                print("5 Error: ", p)
                print("5 Error: ", PyGage.GetErrorString(p))

        # ===== Do Analysis ===================================================
        if work_buffer_active:
            # wait for the thread if it is still held up
            if active_threads[thread_count]:
                work_threads[thread_count].join()
                active_threads[thread_count] = False

            # then start the thread on analyzing new data
            stream_info.WorkBuffer[:] = buffer[:]
            args = (
                loop_count,
                g_cardTotalData,
                stream_info.WorkBuffer,
                mp_values,
                mp_arrays,
                *args_doanalysis,
            )
            work_threads[thread_count] = threading.Thread(target=DoAnalysis, args=args)
            work_threads[thread_count].start()
            active_threads[thread_count] = True

        # ===== continue loop =================================================
        loop_count += 1

        # increment the work buffer
        if buffer_count == len(buffer_list) - 1:
            buffer_count = 0
        else:
            buffer_count += 1

        # increment the thread count after the first loop
        if work_buffer_active:
            if thread_count == len(work_threads) - 1:
                thread_count = 0
            else:
                thread_count += 1

        work_buffer_active = True

        if stream_stop_event.is_set():
            done = True

        print(loop_count)

    PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
    PyGage.FreeStreamingBuffer(handle, card_index, buffer2)
    PyGage.FreeStreamingBuffer(handle, card_index, buffer3)
    PyGage.FreeStreamingBuffer(handle, card_index, buffer4)
    PyGage.FreeSystem(handle)


def DoAnalysis(loop_count, g_cardTotalData, workbuffer, mp_values, mp_arrays, *args):
    """
    Do analysis on stream work buffer

    Args:
        loop_count (int):
            current loop count in the stream while loop
        g_cardTotalData (list):
            list of total data, they make it a list for each card, so you'll
            only have one element in the list
        workbuffer (buffer):
            stream's work buffer. You can retrieve it with np.formbuffer(workbuffer, np.int16)
        mp_values (list of mp.Value):'
            You passed this list to the stream function to be actively updated by
            this DoAnalysis function
        mp_arrays (list of mp.Array):
            You passed this list to the stream function to be actively updated by
            this DoAnalysis function
        args (tuple):
            tuple containing additional arguments you passed to the stream
            function that are needed by this DoAnalysis function
    """
    (mode, *args_remaining) = args
    buffer = np.frombuffer(workbuffer, np.int16)

    if mode == "average":
        (ppifg,) = args_remaining
        N = int(buffer.size // ppifg)
        buffer.resize((N, ppifg))

        (X,) = mp_arrays
        with X.get_lock():
            X[:] = np.sum(buffer, axis=0)

    if mode == "save average":
        (ppifg, savebuffersize, stream_stop_event) = args_remaining
        N = int(buffer.size // ppifg)

        if loop_count * ppifg == savebuffersize:
            stream_stop_event.set()
        if loop_count * ppifg > savebuffersize:
            print("stop flag already set, skipping this one")
            return

        buffer.resize((N, ppifg))
        summed = np.sum(buffer, axis=0)

        (X,) = mp_arrays
        start = (loop_count - 1) * ppifg
        stop = loop_count * ppifg
        with X.get_lock():
            X[start:stop] = summed

    if mode == "save":
        (savebuffersize, stream_stop_event) = args_remaining
        size = buffer.size

        if loop_count * size == savebuffersize:
            stream_stop_event.set()

        if loop_count * size > savebuffersize:
            print("stop flag already set, skipping this one")
            return

        (X,) = mp_arrays
        start = (loop_count - 1) * size
        stop = loop_count * size
        with X.get_lock():
            X[start:stop] = buffer

    (mp_values,) = mp_values
    mp_values.value = g_cardTotalData[0]


buffer_size_to_sample_size = lambda x: x / 2
sample_size_to_buffer_size = lambda x: x * 2

if __name__ == "__main__":
    segmentsize = 77760

    stream_ready_event = mp.Event()
    stream_start_event = mp.Event()
    stream_stop_event = mp.Event()
    stream_error_event = mp.Event()

    N_analysis_threads = 2

    mode = "save"
    N_avg = 500
    N_save = 500
    mp_values = [mp.Value("q")]

    if mode == "average":
        buffersize = sample_size_to_buffer_size(segmentsize * N_avg)
        args_doanalysis = (mode, segmentsize)
        mp_arrays = [mp.Array("q", segmentsize)]

    if mode == "save average":
        buffersize = sample_size_to_buffer_size(segmentsize * N_avg)
        savebuffersize = segmentsize * N_save
        args_doanalysis = (mode, segmentsize, savebuffersize, stream_stop_event)
        mp_arrays = [mp.Array("q", savebuffersize)]

    if mode == "save":
        buffersize = sample_size_to_buffer_size(segmentsize)
        savebuffersize = segmentsize * N_save
        args_doanalysis = (mode, savebuffersize, stream_stop_event)
        mp_arrays = [mp.Array("q", savebuffersize)]

    args = (
        inifile_default,
        buffersize,
        stream_ready_event,
        stream_start_event,
        stream_stop_event,
        stream_error_event,
        N_analysis_threads,
        mp_values,
        mp_arrays,
        args_doanalysis,
    )

    process = mp.Process(target=stream, args=args)
    process.start()
