package dbms.access;

import dbms.memory.AbstractBufferManager;
import dbms.memory.FullBufferException;

public class File extends AbstractFile {
	
	public File(AbstractBufferManager manager, int addr) {
		super(manager, addr);
	}
	
	/* This method prints all records stored on the disk page at address 'addr'
	 * and returns the disk address of the next page in the file.  The page is
	 * retrieved by means of the buffer manager and must be released as soon as
	 * all the records in it have been printed. Records must be printed in the
	 * order in which they are stored on the page.
	 * 
	 * The file, page and record formats are described in the coursework sheet.
	 * 
	 * Once properly parsed, each record will consist of a String of length 10
	 * (right-padded with spaces) and an int between 0 and 65355. Records must
	 * be printed to stdout using the method 'printRecord(String,int)' below to
	 * ensure a uniform output.
	 */
	public int printPage(int addr) throws FullBufferException{
		int nextAddr = 0;
		byte numberSlot = 0;
		int fAddr = manager.fetch(addr);		
		byte[] field1 = new byte[10];
		int field2 = 0;
		byte[] pageContent = new byte[128];
		for(int i = 0; i < 128; i++){
			pageContent[i] = manager.bufferPool[fAddr + i];
		}
		nextAddr = ((pageContent[1]&0xff)<<8 &0xFF00)| (pageContent[0]&0xFF);
		int negative = pageContent[1]&0b10000000;
		if(negative!=0){
			nextAddr = -(65535 - nextAddr + 1);
		}
		numberSlot = pageContent[2];
		
		for(int i = 0; i < numberSlot; i++){
			for(int j = 0; j < 10; j++){
				field1[j] = manager.bufferPool[fAddr + i * 12 + j + 8];
			}
			field2 = (manager.bufferPool[fAddr + i * 12 + 18]&0xFF)|((manager.bufferPool[fAddr + i * 12 + 19]<<8)&0xFF00);
			String s = new String(field1);
			printRecord(s, field2);
		}
		manager.release(fAddr, false);
		return nextAddr;
	}


}
