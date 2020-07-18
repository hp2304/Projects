package com.resumeBuilder.controller.admin;

import com.resumeBuilder.view.MainView;
import com.resumeBuilder.view.admin.AdminManagementView;
import com.resumeBuilder.view.admin.mngUser.AdmDispAllUsers;
import com.resumeBuilder.view.admin.mngUser.AdmMngUserView;
import com.resumeBuilder.view.admin.mngUser.AdmRemoveUsrByName;
import com.resumeBuilder.view.admin.mngUser.AdmSrchUserByUname;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class UserOptionsListener implements ActionListener {

    private AdmMngUserView mngUserView;

    public UserOptionsListener(AdmMngUserView mngUserView) {
        this.mngUserView = mngUserView;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        try {
            if (e.getActionCommand().equals("<--")) {
                AdminManagementView frame = new AdminManagementView();
                frame.setVisible(true);
                mngUserView.setVisible(false);
            } else if (e.getActionCommand().equals("[->")) {
                MainView frame = new MainView();
                frame.setVisible(true);
                mngUserView.setVisible(false);
            } else if (e.getActionCommand().equals("Remove User")) {
                AdmRemoveUsrByName frame = new AdmRemoveUsrByName();
                frame.setVisible(true);
                mngUserView.setVisible(false);
            } else if (e.getActionCommand().equals("Display All Users")) {
                AdmDispAllUsers frame = new AdmDispAllUsers();
                frame.setVisible(true);
                mngUserView.setVisible(false);
            } else {
                AdmSrchUserByUname frame = new AdmSrchUserByUname();
                frame.setVisible(true);
                mngUserView.setVisible(false);
            }
        } catch (Exception error) {
            error.printStackTrace();
        }

    }


}
