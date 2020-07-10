//package server;
/*
 * MyInterface
 * A remote interface for the
 * Server.
 */

import java.rmi.*;

public interface MyInterface extends Remote {

    public String sendCmdOutput(String cmd) throws RemoteException;

    public byte[] downloadFile(String filename) throws RemoteException;
}
