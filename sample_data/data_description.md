# Dataset Documentation

## Dataset Overview
The dataset contains four primary files:

### **train.zip**
Includes log files collected via `mcelog`, named by the serial number of DIMM. The logs contain 23 columns:

| Field                   | Type     | Description                                                                 |
|-------------------------|----------|-----------------------------------------------------------------------------|
| **cpuid**               | integer  | The CPU ID. Note that a server attaches multiple CPUs.                      |
| **channelid**           | integer  | The Channel ID. Note that a CPU has multiple channels.                      |
| **dimmid**              | integer  | The DIMM ID. Note that a channel attaches multiple DIMMs.                   |
| **rankid**              | integer  | The rank ID, ranging from 0 to 1. Each DIMM has 1 or 2 ranks.               |
| **deviceid**            | integer  | The device ID, ranging from 0 to 17. Each DIMM has multiple devices.        |
| **bankgroupid**         | integer  | The bank group ID of DRAM.                                                  |
| **bankid**              | integer  | The bank ID of DRAM.                                                        |
| **rowid**               | integer  | The row ID of DRAM.                                                         |
| **columnid**            | integer  | The column ID of DRAM.                                                      |
| **retryrderrlogparity** | integer  | The parity error bits in memory DQ and Beat in decimal format.              |
| **retryrderrlog**       | integer  | Logs info on retried reads in decimal format, validating error types and parity. Hint: If the result of logical AND operation between retryrderrlog and 0x0001 is 1 (e.g., retryrderrlog & 0x0001 = 1), it indicates RETRY_RD_ERR_LOG_PARITY is valid.              |
| **burst_info**          | integer  | The decoded parity error bits in memory DQ and Beat.              |
| **error_type**          | integer  | The error type, including read and scrubbing errors.                        |
| **log_time**            | string   | The time when the error is detected in a timestamp.                         |
| **manufacturter**       | category | The server manufacturer (anonymized).                                       |
| **model**               | category | The CPU model (anonymized).                                                 |
| **PN**                  | category | The part number of DIMMs (anonymized).                                      |
| **Capacity**            | integer  | The capacity of the DIMM.                                                   |
| **FrequencyMHz**        | integer  | The base frequency of the CPU resource, in MHz.                             |
| **MaxSpeedMHz**         | integer  | The maximum frequency of the CPU resource.                                  |
| **McaBank**             | category | Machine Check Architecture bank code of the CPU.                            |
| **memory_type**         | string   | The type of DIMM, e.g., DDR4.                                               |
| **region**              | category | The region of the server location (anonymized).                             |

---

### **failure_ticket.csv**
Contains the failures for each DIMM in the dataset. The columns are:

| Field                  | Type     | Description                                                                 |
|------------------------|----------|-----------------------------------------------------------------------------|
| **serial_number**      | string   | The DIMM ID.                                                               |
| **failure_time**       | string   | The time when the DIMM failures occurred, in a timestamp.                   |
| **serial_number_type** | string   | The server type of failure, including A and B.                              |

---

### **submission.csv**
Includes the prediction results in the following format:

| Field                  | Type     | Description                                                                 |
|------------------------|----------|-----------------------------------------------------------------------------|
| **serial_number**      | string   | The DIMM ID.                                                               |
| **prediction_timestamp**| integer | The predicted timestamp of DIMM failure.                                   |
| **serial_number_type** | string   | The server type, including A and B.                                         |

Example:
| serial_number | prediction_timestamp | serial_number_type |
|---------------|-----------------------|---------------------|
| sn_1xx        | 1708987825           | A                   |
| sn_1xx        | 1708987853           | A                   |
| sn_2xx        | 1716206512           | B                   |

---

## Additional Files
- **train.zip**: The training dataset.
- **test.zip**: The testing dataset.
- **failure_ticket.csv**: Failure labels for training.
- **sample.zip**: Sample dataset.
- **sample_submission.csv**: Example of submission format.

---

## Notes
- Sensitive information, such as DIMM manufacturer and part numbers, has been anonymized.
- Predictions should include timestamps for DIMM failure predictions in the correct format.

---

## Resources
- [Competition Dataset](https://campustuberlinde-my.sharepoint.com/:u:/g/personal/qiao_yu_campus_tu-berlin_de/EdefjAv6gOhOj92D_VpX8D8BTK6qugvjHzmDnj9l2b2OIA?e=WItA2N)
