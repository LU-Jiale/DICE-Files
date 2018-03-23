package dbms;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import dbms.access.File;
import dbms.memory.AbstractBufferManager.Policy;
import dbms.memory.BufferManager;
import dbms.memory.DiskSpaceManager;

public class FileTest {

	public static void main(String[] args) throws IOException, Exception {
		Path p = Paths.get(args[0]);
		byte[] data = Files.readAllBytes(p);
		DiskSpaceManager diskMan = new DiskSpaceManager(data);
		Policy policy;
		switch (args[1]) {
		case "FIFO":
			policy = Policy.FIFO;
			break;
		case "LIFO":
			policy = Policy.LIFO;
			break;
		case "LRU":
			policy = Policy.LRU;
			break;
		case "MRU":
			policy = Policy.MRU;
			break;
		default:	
			throw new Exception("Unknown replacement policy '" + args[1] + "'");
		}
		int frames = Integer.parseInt(args[2]);
		if (frames < 0) {
			throw new Exception("Negative number of frames (seriously?)");
		}
		if (frames == 0) {
			throw new Exception("The buffer pool must have some frames!");
		}
		BufferManager manager = new BufferManager(policy, frames, 128, diskMan);
		File sc = new File(manager, Integer.parseInt(args[3]));
		sc.printAll();
	}
}
