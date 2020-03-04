# 6.824 2020 Lecture 4: **Primary/Backup Replication**

**`Today:`**

- Primary/Backup Replication for Fault Tolerance
- Case study of VMware FT, an extreme version of the idea

The _topic_ is (still) **fault tolerance** to provide availability despite server and network failures using replication.

**`What kinds of **failures** can replication deal with?`**

- **"fail-stop"** failure of a single replica
  - fan stops working, CPU overheats and shuts itself down
  - someone trips over replica's power cord or network cable
  - software notices it is out of disk space and stops
- Maybe not defects in h/w or bugs in s/w or human configuration errors
  - Often not fail-stop
  - May be **correlated** (i.e. cause all replicas to crash at the same time)
  - But, sometimes _can be detected_ (e.g. **checksums**)
- How about earthquake or city-wide power failure?
  - Only if replicas are physically separated

**`Is replication worth the Nx expense?`**

## Replication

Two main replication approaches:

- **State transfer** -> memory
  - Primary replica executes the service
  - Primary sends [new] state to backups
- **Replicated state machine** -> primary backup op

  - Clients send operations to primary,
    - primary sequences and sends to backups
  - All replicas execute all operations
  - If same start state,
    - same operations,
    - same order,
    - deterministic,
    - then same end state.

- State transfer is **simpler** but state **may be large**, slow to transfer over network
- Replicated state machine often generates **less network traffic**
- Operations are often small compared to state but complex to get right

VM-FT uses replicated state machine, as do Labs 2/3/4

Multi-core

### Big Questions

- What state to replicate?
- Does primary have to wait for backup?
- When to cut over to backup?
- Are anomalies visible at cut-over?
- How to bring a replacement backup up to speed?

### **At what level do we want replicas to be identical?**

- **Application** state, e.g. a database's tables?
  - GFS works this way
  - Can be _efficient_; primary only sends high-level operations to backup
  - Application code (server) must _understand fault tolerance_, to e.g. _forward op stream_
- **Machine level**, e.g. registers and RAM content?
  - might allow us to replicate any existing server **w/o modification**!
  - requires **forwarding of machine events** (_interrupts, DMA, &c_)
  - requires "machine" modifications to send/recv event stream...

Today's paper (**VMware FT**) replicates _machine-level_ state

**Transparent**: can run any existing O/S and server software!  
Appears like a single server to clients

## Overview

[diagram: app, O/S, VM-FT underneath, disk server, network, clients]

- words:
  - **hypervisor == monitor == VMM** (virtual machine monitor)
  - O/S+app is the **"guest"** running inside a virtual machine
- two machines, primary and backup
- primary sends all external events (client packets &c) to backup over network
  - "logging channel", carrying log entries
- ordinarily, backup's output is suppressed by FT
- if either stops being able to talk to the other over the network
  - "goes live" and provides sole service
  - if primary goes live, it stops sending log entries to the backup

### VMM emulates a local disk interface

- but actual storage is on a network server
- treated much like a client:
  - usually only **primary communicates** with disk server (backup's FT discards)
  - if backup goes live, it talks to disk server
- external disk makes creating a new backup faster (don't have to copy primary's disk)

### When does the primary have to send information to the backup?

Any time something happens that might cause their executions to **diverge**.  
Anything that's **not a deterministic** consequence of executing instructions.

Local Area Network

### What sources of divergence must FT handle?

- Most instructions execute identically on **primary** and **backup**.
- As long as _memory+registers_ are identical,
- which we're assuming by induction.
- Inputs from external world -- just network packets.
- These appear as DMA'd data plus an interrupt.
- Timing of interrupts.
- Instructions that aren't functions of state, such as reading current time.
- Not **multi-core** races, since uniprocessor only.

### Why would divergence be a disaster?

b/c state on backup would **differ from** state on primary, and if primary then failed, clients would see **inconsistency**.

Example: GFS lease expiration

- Imagine we're replicating the GFS master
- Chunkserver must send **"please renew" **msg before 60-second lease expires
- Clock interrupt drives master's notion of time
- Suppose chunkserver sends "please renew" just around 60 seconds
- On primary, clock interrupt happens just after request arrives.
  - Primary copy of master renews the lease, to the same chunkserver.
- On backup, clock interrupt happens just before request.
  - Backup copy of master expires the lease.
- If primary fails, backup takes over, it will think there is no lease, and grant it to a different chunkserver.
- Then two chunkservers will have lease for same chunk.

So: backup must see same events, in same order, at same points in instruction stream.

Each log entry: **instruction #, type, data**.

### FT's handling of **timer interrupts**

Goal: **primary and backup should see interrupt at the same point in the instruction stream**

**Primary:**

- FT fields the timer interrupt
- FT reads instruction number from CPU
- FT sends "timer interrupt at instruction X" on logging channel
- FT delivers interrupt to primary, and resumes it
- (this relies on CPU support to interrupt after the X'th instruction)

**Backup:**

- ignores its own timer hardware
- FT sees log entry _before_ backup gets to instruction X
- FT tells CPU to interrupt (to FT) at instruction X
- FT mimics a timer interrupt to backup

FT's handling of network packet arrival (input)

**Primary:**

- FT tells NIC to copy packet data into FT's private **"bounce buffer"**
- At some point NIC does DMA, then interrupts
- FT gets the interrupt
- FT pauses the primary
- FT copies the bounce buffer into the primary's memory
- FT simulates a NIC interrupt in primary
- FT sends the packet data and the instruction # to the backup

**Backup:**

FT gets data and instruction # from log stream
FT tells CPU to interrupt (to FT) at instruction X
FT copies the data to backup memory, simulates NIC interrupt in backup

### Why the bounce buffer?

We want the data to appear in memory at exactly the same point in execution of the primary and backup.
Otherwise they may diverge.

### Note that the backup must lag by one one log entry

- Suppose primary gets an interrupt, or input, after instruction X
- If backup has already executed past X, it cannot handle the input correctly
- So backup FT can't start executing at all until it sees the first log entry
  - Then it executes just to the instruction # in that log entry
  - And waits for the next log entry before resuming backup

**Example:** non-deterministic instructions

- some instructions yield different results even if primary/backup have same state
  - e.g. reading the current time or cycle count or processor serial #
- Primary:
  - FT sets up the CPU to interrupt if primary executes such an instruction
  - FT executes the instruction and records the result
  - sends result and instruction # to backup
- Backup:
- FT reads log entry, sets up for interrupt at instruction #
- FT then supplies value that the primary got

### What about output (sending network packets)?

- Primary and backup both execute instructions for output
- Primary's FT actually does the output
- Backup's FT discards the output

**Output example:** DB server

- clients can send "increment" request
  - DB increments stored value, replies with new value
- so:
  - suppose the server's value starts out at 10
  - network delivers client request to FT on primary
  - primary's FT sends on logging channel to backup
  - FTs deliver request to primary and backup
  - primary executes, sets value to 11, sends "11" reply, FT really sends reply
  - backup executes, sets value to 11, sends "11" reply, and FT discards
  - the client gets one "11" response, as expected

But wait:

- suppose primary crashes just after sending the reply
  - so client got the "11" reply
- AND the logging channel discards the log entry w/ client request
  - primary is dead, so it won't re-send
- backup goes live
  - but it has value "10" in its memory!
- now a client sends another increment request
  - it will get "11" again, not "12"
- oops

**Solution**: the Output Rule (Section 2.2)

before primary sends output, must wait for backup to acknowledge all previous log entries

Again, with output rule:

- **primary:**
  - receives client "increment" request
  - sends client request on logging channel
  - about to send "11" reply to client
  - first waits for backup to acknowledge previous log entry
  - then sends "11" reply to client
- suppose the primary crashes at some point in this sequence
- if before primary receives acknowledgement from backup
  - maybe backup didn't see client's request, and didn't increment
  - but also primary won't have replied
- if after primary receives acknowledgement from backup
  - then client may see "11" reply
  - but backup guaranteed to have received log entry w/ client's request
  - so backup will increment to 11

### The Output Rule is a big deal

- Occurs in some form in all replication systems
- A serious constraint on performance
- An area for application-specific cleverness
  - Eg. maybe no need for primary to wait before replying to read-only operation
- FT has no application-level knowledge, must be conservative

### Q: What if the primary crashes just after getting ACK from backup, but before the primary emits the output? Does this mean that the output won't ever be generated?

A: Here's what happens when the primary fails and the backup goes live.  
The backup got some log entries from the primary.  
The backup continues executing those log entries WITH OUTPUT DISCARDED.  
After the last log entry, the backup goes live -- stops discarding output  
In our example, the last log entry is arrival of client request  
So after client request arrives, the client will start emitting outputs  
And thus it will emit the reply to the client

### Q: But what if the primary crashed _after_ emitting the output? Will the backup emit the output a _second_ time?

A: Yes.  
OK for TCP, since receivers ignore duplicate sequence numbers.  
OK for writes to disk, since backup will write same data to same block #.

Duplicate output at cut-over is pretty common in replication systems  
Clients need to keep enough state to ignore duplicates  
Or be designed so that duplicates are harmless

### Q: Does FT cope with network partition -- could it suffer from split brain? E.g. if primary and backup both think the other is down. Will they both go live?

A: The disk server breaks the tie.  
Disk server supports **atomic test-and-set**.  
If primary or backup thinks other is dead, attempts test-and-set.  
If only one is alive, it will win test-and-set and go live.  
If both try, one will lose, and halt.

The disk server may be a single point of failure  
If disk server is down, service is down  
They probably have in mind a replicated disk server

### Q: Why don't they support multi-core?

Performance (table 1)

- FT/Non-FT: impressive!
  - little slow down
- Logging bandwidth
  - Directly reflects disk read rate + network input rate
  - 18 Mbit/s for my-sql
- These numbers seem low to me
  - Applications can read a disk at at least 400 megabits/second
  - So their applications aren't very disk-intensive

### When might FT be attractive?

Critical but low-intensity services, e.g. name server.  
Services whose software is not convenient to modify.

### What about replication for high-throughput services?

- People use application-level replicated state machines for e.g. databases.
  - The state is just the DB, not all of memory+disk.
  - The events are DB commands (put or get), not packets and interrupts.
- Result: less fine-grained synchronization, less overhead.
- GFS use application-level replication, as do Lab 2 &c

## Summary:

- Primary-backup replication
  - VM-FT: clean example
- How to cope with partition without single point of failure?
  - Next lecture
  - How to get better performance?
  - Application-level replicated state machines
- Test-And-Set

---

VMware KB (#1013428) talks about multi-CPU support.  
VM-FT may have switched from a replicated state machine approach to the state transfer approach, but unclear whether that is true or not.

http://www.wooditwork.com/2014/08/26/whats-new-vsphere-6-0-fault-tolerance/

http://www-mount.ece.umn.edu/~jjyi/MoBS/2007/program/01C-Xu.pdf

## QA

### Q: The introduction says that it is more difficult to ensure deterministic execution on physical servers than on VMs. Why is this the case?

A: **Ensuring determinism** is easier on a VM because the hypervisor emulates and controls many aspects of the hardware that might differ between primary and backup executions, for example the precise timing of interrupt delivery.

### Q: What is a hypervisor?

A: A hypervisor is part of a Virtual Machine system; it's the same as the **Virtual Machine Monitor** (VMM).

The hypervisor emulates a computer, and a guest operating system (and applications) execute inside the
emulated computer.
The emulation in which the guest runs is often called the virtual machine.
In this paper, the primary and backup are guests running inside virtual machines, and FT is part of the
hypervisor implementing each virtual machine.

### Q: Both GFS and VMware FT provide **fault tolerance**. How should we think about when one or the other is better?

A: FT **replicates computation**;
you can use it to transparently add fault-tolerance to any existing network server.
FT provides fairly **strict consistency** and is transparent to server and client.
You might use FT to make an existing **mail server fault-tolerant**, for example.

GFS, in contrast, provides fault-tolerance just for **storage**.
Because GFS is specialized to a **specific simple service (storage)**, its replication is more efficient than FT.
For example, GFS does not need to cause interrupts to happen at exactly the same instruction on all
replicas.
GFS is usually only one piece of a larger system to implement complete fault-tolerant services.

For example, VMware FT itself relies on a fault-tolerant storage service shared by primary and backup (the Shared Disk in Figure 1), which you could use something like GFS to implement (though at a detailed level GFS wouldn't be quite the right thing for FT).

### Q: How do Section 3.4's bounce buffers help avoid **races**?

A: The problem arises when a network packet or requested disk block arrives at the primary and needs to be copied into the primary's memory.
Without FT, the relevant hardware copies the data into memory while software is executing.
Guest instructions could read that memory during the DMA; depending on exact timing, the guest might see or not see the DMA'd data (this is the race).
It would be bad if the primary and backup both did this, but due to slight timing differences one read just after the DMA and the other just before.
In that case they would diverge.

FT avoids this problem by not copying into **guest memory** while the primary or backup is executing.
FT first copies the network packet or disk block into a private **"bounce buffer"** that the primary cannot access.
When this first copy completes, the FT hypervisor interrupts the primary so that it is not executing. FT records the **point at which it interrupted the primary** (as with any interrupt).
Then FT **copies the bounce buffer into the primary's memory**, and after that allows the primary to continue executing.
FT sends the data to the backup on the log channel.
The backup's FT interrupts the backup at the same instruction as the primary was interrupted, copies the data into the backup's memory while the backup is into executing, and then resumes the backup.

The effect is that the network packet or disk block appears at exactly the same time in the primary and backup, so that no matter when they read the memory, both see the same data.

### Q: What is **"an atomic test-and-set** operation on the shared storage"?

A: The system uses a network disk server, shared by both primary and backup (the "shared disk" in Figure 1).
That network disk server has a **"test-and-set service"**.
The test-and-set service maintains a flag that
is initially set to false.
If the primary or backup thinks the other server is dead, and thus that it should take over by itself, it first sends a test-and-set operation to the disk server.
The server executes roughly this code:

```go
test-and-set() {
  acquire_lock()
  if flag == true:
    release_lock()
    return false
  else:
    flag = true
    release_lock()
    return true
}
```

The primary (or backup) only takes over ("goes live") if test-and-set returns true.

The higher-level view is that, if the primary and backup lose network contact with each other, we **want only one of them to go live**.
The danger is that, if both are up and the network has failed, both may go live and develop split brain.
If only one of the primary or backup can talk to
the disk server, then that server alone will go live.
But what if both can talk to the disk server?
Then the network disk server acts as a tie-breaker; test-and-set returns true only to the first call.

### Q: How much performance is lost by following the Output Rule?

A: Table 2 provides some insight.
By following the output rule, the transmit rate is reduced, but not hugely.

### Q: What if the application calls a **random number** generator? Won't that yield different results on primary and backup and cause the executions to diverge?

A: The primary and backup will get **the same number from their random number generators**.
All the sources of randomness are controlled by the hypervisor.
For example, the application **may use** the current time, or a hardware cycle counter, or precise interrupt times **as sources of randomness**.
In all three cases the hypervisor intercepts the relevant instructions on both primary and backup and ensures they produce the same values.

### Q: How were the creators certain that they captured all possible forms of non-determinism?

A: My guess is as follows.
The authors work at a company where many people understand VM hypervisors, microprocessors, and internals of guest OSes well, and will be aware of many of the pitfalls.
For VM-FT specifically, the authors leverage the log and replay support from a previous a project (deterministic replay), which must have already dealt with sources of non-determinism.
I assume the designers of deterministic replay did extensive testing and gained experience with sources of non-determinism that the authors of VM-FT use.

### Q: What happens if the primary fails just after it sends output to the external world?

A: The backup will likely **repeat** the output after taking over, so that it's generated **twice**.
This duplication is not a problem for network and disk I/O.
If the output is a network packet, then the receiving client's **TCP** software will **discard** the duplicate automatically.
If the output event is a **disk I/O**, disk I/Os are **idempotent** (both write the same data to the same location, and there are no intervening I/Os).

### Q: Section 3.4 talks about disk I/Os that are outstanding on the primary when a failure happens; it says "Instead, we re-issue the pending I/Os during the go-live process of the backup VM." Where are the **pending I/Os** located/stored, and how far back does the re-issuing need to go?

A: The paper is talking about disk I/Os for which there is **a log entry** indicating the I/O was started but no entry indicating completion.
These are the I/O operations that must be re-started on the backup.
When an I/O completes, the I/O device generates an I/O completion interrupt.
So, if the I/O completion interrupt is missing in the log, then the backup restarts the I/O.
If there is an I/O completion interrupt in the log, then there is no need to restart the I/O.

### Q: How is the backup FT able to deliver an interrupt at a particular point in the backup instruction stream (i.e. at the same instruction at which the interrupt originally occured on the primary)?

A: Many CPUs support a feature (the "**performance counters")** that lets the FT VMM tell the CPU a number of instructions, and the CPU will interrupt to the FT VMM **after that number of instructions**.

### Q: How secure is this system?

A: The authors assume that the primary and backup follow the protocol and are not malicious (e.g., an attacker didn't compromise the hypervisors).
The system cannot handle **compromised hypervisors**.
On the other hand, the hypervisor can probably defend itself against malicious or buggy guest operating systems and applications.

### Q: Is it reasonable to address only the **fail-stop failures**? What are other type of failures?

A: It is reasonable, since many real-world failures are essentially fail-stop, for example many network and power failures.
Doing better than this requires coping with computers that appear to be operating correctly but actually compute incorrect results; in the worst case,
perhaps the failure is the result of a malicious attacker.
This larger class of non-fail-stop failures is often called **"Byzantine"**.
There are ways to deal with Byzantine failures, which we'll touch on at the end of the course, but most of 6.824 is about fail-stop failures.
