package com.resumeBuilder.controller.user;

import com.resumeBuilder.model.user.*;
import com.resumeBuilder.view.MainView;
import com.resumeBuilder.view.user.UserAccountCreation;
import com.resumeBuilder.view.user.UserEditOptions;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class AccCreationListener implements ActionListener {

    private UserAccountCreation accCreationView;
    private UserProjectDtls prj;
    private ArrayList<UserEduDtls> ref;
    private UserOtherDtls other;
    private UserPersonalDtls personal;

    public AccCreationListener(UserAccountCreation view) {
        accCreationView = view;
        ref = new ArrayList<>();
        other = new UserOtherDtls("<None>", "<None>", "<None>", "<None>");
        personal = new UserPersonalDtls("DD/MM/YYYY", "<None>", "<None>", "<None>", "<None>", "<None>", "<None>", "<None>");
        prj = new UserProjectDtls("<None>", "<None>", "0");
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        if (e.getActionCommand().equals("<--")) {
            MainView mainView = new MainView();
            mainView.setVisible(true);
            accCreationView.setVisible(false);
        } else {
            try {
                if (accCreationView.getTxtPwd().trim().equals("") || accCreationView.getTxtConPwd().trim().equals("") || accCreationView.getTxtUserName().trim().equals("") || accCreationView.getTxtEmail().trim().equals("") || accCreationView.getTxtFirstName().trim().equals("") || accCreationView.getTxtLastName().trim().equals("")) {
                    throw new Exception("Not Valid");
                } else if (accCreationView.getTxtConPwd().trim().equals("") && accCreationView.getTxtPwd().trim().equals("") && accCreationView.getTxtUserName().trim().equals("") && accCreationView.getTxtEmail().trim().equals("") && accCreationView.getTxtFirstName().trim().equals("") && accCreationView.getTxtLastName().trim().equals("")) {
                    throw new Exception("Not Valid");
                } else {
                    int flag = 0;
                    ArrayList<User> user = new ArrayList<>();
                    user = UserStorage.readUsers();
                    if (user != null) {
                        for (User temp : user) {
                            if (accCreationView.getTxtUserName().equals(temp.getUsrUserName()) || accCreationView.getTxtEmail().equals(temp.getUsrEmailId())) {
                                JOptionPane.showMessageDialog(null, "This UserName or EmailId is used already please use different");
                                flag = 1;
                            }
                        }
                    }
                    if (flag == 0) {
                        if (accCreationView.getTxtConPwd().equals(accCreationView.getTxtPwd())) {
                            JOptionPane.showMessageDialog(null, "YOUR ACCOUNT CREATED SUCCESSFULLY");
                            User newUser = new User(accCreationView.getTxtUserName(), accCreationView.getTxtPwd(), accCreationView.getTxtFirstName(), accCreationView.getTxtLastName(), accCreationView.getTxtEmail(), other, personal, ref, prj);
                            UserStorage.writeUserFile(newUser);
                            UserEditOptions edit = new UserEditOptions(newUser);
                            edit.setVisible(true);
                            accCreationView.setVisible(false);
                        } else {
                            JOptionPane.showMessageDialog(null, "Password and Confirm_Password is not the same");
                        }
                    }
                }
            } catch (Exception err) {
                JOptionPane.showMessageDialog(null, "Fill out all details", "ERROR IN CREATING ACCOUNT", JOptionPane.INFORMATION_MESSAGE);
            }
        }
    }

}


