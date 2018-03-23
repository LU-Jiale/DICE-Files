package dbms.memory;

import java.util.ArrayList;
import java.util.ListIterator;

public abstract class AbstractBufferManager {

	// Supported replacement policies
	public static enum Policy { LRU, MRU, FIFO, LIFO };

	// The buffer pool is an array of bytes (like a disk, but in main memory)
	public final byte[] bufferPool;

	// The size of a page (in bytes)
	public final int pageSize;

	// The specific replacement policy used by this buffer manager
	protected final Policy replPolicy;

	/* The bookkeeping table is a list of FrameInfo objects, where the order is
	 * determined by the replacement policy in use.
	 * Independently of the policy, a page read from disk will be loaded into
	 * the first *empty* frame in the list; if there is none, the first frame
	 * in the list with pin count 0 will be selected for replacement.
	 */
	protected final ArrayList<FrameInfo> bookkeeping = new ArrayList<FrameInfo>();

	// The underlying disk space manager used for I/O.
	protected final DiskSpaceManager spaceManager;

	/* The buffer manager is initialized with a replacement policy, the number
	 * of frames in the buffer pool, the page size, and a disk space manager.
	 * None of these can be changed once the buffer manager is instantiated.
	 * The buffer pool is initialized with blank data, split into 'numFrames'
	 * frames; the bookkeeping list is filled with FrameInfo objects initially
	 * ordered by ascending address of the corresponding frame.
	 */
	public AbstractBufferManager(Policy replPolicy, int numFrames,
			int pageSize, DiskSpaceManager spaceMan) {
		this.replPolicy = replPolicy;
		this.pageSize = pageSize;
		this.spaceManager = spaceMan;
		bufferPool = new byte[pageSize * numFrames];
		for ( int i = 0; i * pageSize < bufferPool.length; i++ ) {
			bookkeeping.add(new FrameInfo(i * pageSize));
		}
	}

	// Prints out the information held in the bookkeeping table.
	public final void bookkeepingInfo() {
		ListIterator<FrameInfo> it = bookkeeping.listIterator();
		while (it.hasNext()) {
			System.out.println(it.next());
		}
	}

	/* Prints out the buffer pool in hexadecimal format, in rows of 16 bytes
	 * and enclosing each byte in square brackets. Useful for testing (when
	 * the buffer pool is <= 10KB in size).
	 */
	public final void poolHexDump() {
		for (int i=0; i < bufferPool.length; i++) {
			if ((i > 0) && (i % 16 == 0)) {
				System.out.println();
			}
			System.out.print(String.format("[%2S]",
					Integer.toHexString(bufferPool[i] & 0xFF)).replace(' ', '0'));
		}
		System.out.println();
	}

	/* This method fetches a page with *disk* address 'pageAddr'. If the page
	 * is not in the buffer pool already, it must be retrieved from disk. The
	 * bookkeeping information must be updated accordingly and by taking into
	 * account the replacement policy in use. Remember to take care of dirty
	 * pages before replacing them.
	 */
	public abstract int fetch(int pageAddr) throws FullBufferException;

	/* This method releases a page, indicating whether it has been modified or
	 * not. The bookkeeping information must be updated accordingly (but it is
	 * independent of the replacement policy in use).
	 */
	public abstract void release(int frameAddr, boolean modified);
}
