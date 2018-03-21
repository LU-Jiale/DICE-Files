package dbms;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;

import dbms.memory.AbstractBufferManager.Policy;
import dbms.memory.BufferManager;
import dbms.memory.DiskSpaceManager;
import dbms.memory.FullBufferException;

public class BufferManagerTest {

	public static void main(String[] args) throws Exception {
		Path p = Paths.get(args[0]);
		byte[] data = Files.readAllBytes(p);
		DiskSpaceManager diskMan = new DiskSpaceManager(data);

		Policy policy = parsePolicy(args[1]);
		int frames = parsePositiveInt(args[2]);
		int pSize = Integer.parseInt(args[3]);
		BufferManager manager = new BufferManager(policy, frames, pSize, diskMan);

		String fetch = "fetch(%d) ==> %s";
		Scanner s = new Scanner(new File(args[4]));
		while (s.hasNext()){
			String[] line = s.next().split(",");
			if (line[0].equals("fetch")) {
				int pAddr = Integer.parseInt(line[1]);
				try {
					int fAddr = manager.fetch(pAddr);
					if (line[2].equals("null")) {
						String err = String.format("ERROR: expected full buffer pool, got frame address %d instead", fAddr);
						System.out.println(String.format(fetch, pAddr, err));
					} else {
						int fAddrExp = Integer.parseInt(line[2]);
						if (fAddr != fAddrExp) {
							String err = String.format("ERROR: expected frame address %d, got %d", fAddrExp, fAddr); 
							System.out.println(String.format(fetch, pAddr, err));
						}
					}
				} catch (FullBufferException e) {
					if (line[2].equals("null") == false) {
						int fAddrExp = Integer.parseInt(line[2]);
						String err = String.format("ERROR: expected frame address %d, but buffer pool is full", fAddrExp);
						System.out.println(String.format(fetch, pAddr, err));
					}
				}
			} else {
				int fAddr = Integer.parseInt(line[1]);
				boolean mod = Boolean.parseBoolean(line[2]);
				if (mod == true) {
					for (int i = 1; i < pSize - 1; i++) {
						manager.bufferPool[fAddr+i] = manager.bufferPool[fAddr+i+1];
					}
					manager.bufferPool[fAddr] = manager.bufferPool[fAddr + pSize - 1];
				}
				manager.release(fAddr, mod);
			}
		}
		s.close();

		System.out.println();
		System.out.println("=== BUFFER POOL ================================================");
		manager.poolHexDump();
		System.out.println();
		System.out.println("=== BOOKKEEPING INFO ===========================================");
		manager.bookkeepingInfo();
		System.out.println();
		System.out.println("=== DISK =======================================================");
		diskMan.diskHexDump();
	}

	public static Policy parsePolicy(String s) throws Exception {
		Policy p;
		switch (s) {
		case "FIFO":
			p = Policy.FIFO;
			break;
		case "LIFO":
			p = Policy.LIFO;
			break;
		case "LRU":
			p = Policy.LRU;
			break;
		case "MRU":
			p = Policy.MRU;
			break;
		default:	
			throw new Exception("Unknown replacement policy '" + s + "'");
		}
		return p;
	}

	public static int parsePositiveInt(String s) throws Exception {
		int k = Integer.parseInt(s);
		if (k <= 0) {
			throw new Exception(String.format("Non-positive number (%d)",k));
		}
		return k;
	}
}
