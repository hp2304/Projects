package com.resumeBuilder.controller.user;

import com.resumeBuilder.model.user.User;

import java.util.ArrayList;

public abstract class FindUser {

    public static User findUser(ArrayList<User> user, String userName, String passWord) {

        for (User temp : user) {
            if (temp.getUsrUserName().equals(userName) && temp.getUsrPassWord().equals(passWord)) {
                return temp;
            }
        }
        return null;
    }

    public static User findUserName(ArrayList<User> user, String userName) {
        for (User temp : user) {

            if (temp.getUsrUserName().equals(userName)) {
                return temp;

            }
        }
        return null;
    }
}
