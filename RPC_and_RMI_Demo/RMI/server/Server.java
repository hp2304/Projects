//package server;
/* Server
 * A server with current weather information.
 * The client can communicate with RMI.
 */

import java.rmi.*;
import java.rmi.registry.*;
import java.rmi.server.*;
import java.net.*;
import java.io.*;

public class Server extends UnicastRemoteObject implements MyInterface{
    public Server() throws RemoteException{};
    private static final long serialVersionUID = 7526472295622776147L;
    public String pwd = new File(".").getAbsolutePath();

    public String sendCmdOutput(String cmd)throws RemoteException{
        String out = "";
        Process proc = null;
        BufferedReader stdResponse;

        if(cmd.split(" ", 2)[0].equals("cd")){
            this.pwd = changePWD(cmd, this.pwd);
            return this.pwd;
        }
        try{
            proc = Runtime.getRuntime().exec(new String[] {"sh", "-c", cmd}, null, new File(this.pwd));
            String s = null;

            if(proc.waitFor()==0){
                stdResponse = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            }
            else{
                stdResponse = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
            }
            while ((s = stdResponse.readLine()) != null) {
                out += s;
                out += "\n";
            }
            stdResponse.close();
            proc.destroy();
        }
        catch(Exception re){
            re.printStackTrace();
        }
        return out;
    }

    private static String changePWD(String cmd, String pwd){
        String s = null, out = "";
        cmd = cmd+";pwd";
        Process proc;
        BufferedReader stdResponse = null;
        try{
            proc = Runtime.getRuntime().exec(new String[] {"sh", "-c", cmd}, null, new File(pwd));
            if(proc.waitFor()==0){
                stdResponse = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            }
            else{
                System.out.println(cmd + " execution failed.");
            }
            while ((s = stdResponse.readLine()) != null) {
                out += s;
            }
            stdResponse.close();
            proc.destroy();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        //this.pwd = out;
        return out;
    }

    public byte[] downloadFile(String filename)throws RemoteException{
        //filename = this.pwd + "/" + filename;
        System.out.println(this.pwd);
        System.out.println(filename);
        File file = new File(this.pwd + "/" + filename.replace("\"", ""));
        Process proc = null;
        try {
            if(file.exists()){
                //String progPath = new File(".").getAbsolutePath();
                //String zipPath = progPath.substring(0, progPath.length()-1) + "buffer.zip";
                String zipPath = this.pwd + "/buffer.zip";
                System.out.println("zip -r " + "buffer.zip " + filename);
                //System.out.println(progPath);
                System.out.println(zipPath);
                
                proc = Runtime.getRuntime().exec(new String[] {"sh", "-c", "zip -r " + "buffer.zip " + filename}, null, new File(this.pwd));
                if(proc.waitFor() != 0){
                    System.out.println("cmd 1 failed...");
                    return null;
                }
                
                File zipfile = new File(zipPath);
                if(zipfile.exists()){
                    byte buffer[] = new byte[(int)zipfile.length()];
                    BufferedInputStream input = new BufferedInputStream(new FileInputStream(zipPath));
                    input.read(buffer,0,buffer.length);
                    input.close();
                    proc = Runtime.getRuntime().exec(new String[] {"sh", "-c", "rm buffer.zip"}, null, new File(this.pwd));
                    //proc = Runtime.getRuntime().exec("rm buffer.zip");
                    if(proc.waitFor() != 0){
                        System.out.println("cmd 2 failed...");
                    }
                    proc.destroy();
                    return buffer;
                }
            }
        }catch(Exception e){
            e.printStackTrace();
        }
        return null;
    }

    // main-method to start the server.
    public static void main(String args[]) {
        try {
            //System.setProperty("java.rmi.server.hostname", "10.20.132.85");
            Server ws  = new Server();
            LocateRegistry.createRegistry(4711);
            Naming.rebind("//localhost:4711/Server", ws);
        
        } catch(RemoteException re) {
            re.printStackTrace();
            System.out.println("Could not start up server.");
        } catch(MalformedURLException mue) {
            System.out.println("The URL was malformed.");
        }
    }
}

