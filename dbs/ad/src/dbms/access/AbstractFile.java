package dbms.access;

import dbms.memory.AbstractBufferManager;
import dbms.memory.FullBufferException;

public abstract class AbstractFile {

	protected final int pageSize;
	protected final int baseAddr;
	protected AbstractBufferManager manager;

	public AbstractFile(AbstractBufferManager manager, int addr) {
		this.manager = manager;
		this.pageSize = manager.pageSize;
		this.baseAddr = addr;
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
	public abstract int printPage(int addr) throws FullBufferException;

	/* Prints a record to stdout in a specific format. Use *only* this method
	 * to print records in your implementation of 'int printPage(int)' above.
	 */
	public final void printRecord(String field1, int field2) {
		System.out.println(field1 + " | " + String.format("%5d", field2));
	}

	/* If your implementation of 'int printPage(int)' is correct, this method
	 * will print the contents of the two-column table the file stores.
	 */
	public final void printAll() {
		int addr = baseAddr;
		while (addr >= 0) {
			try {
				addr = printPage(addr);
			} catch (FullBufferException e) {
				e.printStackTrace();
				break;
			}
		}
	}
}
