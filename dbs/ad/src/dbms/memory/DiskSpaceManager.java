package dbms.memory;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/* This class simulates access to disk (we are still in main memory; I am not
 * going to write my own low-level disk manager!). The disk itself is an array
 * of bytes (the maximum size is about 2GB), and each index of the array is a
 * disk memory address.
 */
public class DiskSpaceManager {

	private final byte[] disk;

	/* The disk is initialized with a fixed 'diskSize' (in bytes) and random
	 * or blank data, depending on whether the value of 'randomData' is true
	 * or false, respectively.
	 */
	public DiskSpaceManager(int diskSize, boolean randomData) {
		disk = new byte[diskSize];
		if (randomData) {
			ThreadLocalRandom.current().nextBytes(disk);
		}
	}

	/* This additional constructor allows one to initialize the disk using an
	 * existing byte array (I will provide some binary dumps for testing).
	 */
	public DiskSpaceManager(byte[] diskDump) {
		this.disk = diskDump;
	}

	/* This method reads 'length' bytes (the size of a page) from disk starting
	 * at address 'addr', and returns them to the caller (the buffer manager).
	 * In our simulation, this causes a disk I/O.
	 */
	byte[] read(int addr, int length) {
		return Arrays.copyOfRange(disk, addr, addr+length);
	}

	/* This method writes a page of bytes to disk at address 'addr'.
	 * In our simulation, this also causes a disk I/O.
	 */
	void write(int addr, byte[] page) {
		for ( int i=0; i < page.length; i++ ) {
			disk[addr+i] = page[i];
		}
	}

	/* Prints out the disk in hexadecimal format, in rows of 16 bytes
	 * and enclosing each byte in square brackets. Useful for testing
	 * (but only with disk sizes <= 10KB).
	 */
	public final void diskHexDump() {
		for (int i=0; i < disk.length; i++) {
			if ((i > 0) && (i % 16 == 0)) {
				System.out.println();
			}
			System.out.print(String.format("[%2S]",
					Integer.toHexString(disk[i] & 0xFF)).replace(' ', '0'));
		}
		System.out.println();
	}
}
